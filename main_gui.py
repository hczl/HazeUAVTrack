import sys
import os
import glob
import time
import math
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
# No need for tqdm in a GUI thread

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QPushButton, QSlider, QFileDialog,
                             QStatusBar, QSizePolicy) # Import QSizePolicy
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QMutex # Import QMutex for thread-safe access

# --- Import your model and utils ---
# Make sure your project structure allows importing these
try:
    from utils.config import load_config
    from utils.create import create_model # This should create the FSDT model
    # Assuming your model class has methods like:
    # __init__(cfg)
    # load_model() # Loads weights
    # dehaze(input_tensor) -> dehazed_tensor
    # predict(input_tensor) -> list of detections [x1, y1, x2, y2, conf, ...]
except ImportError as e:
    print(f"Error importing model utilities: {e}")
    print("Please ensure utils/config.py and utils/create.py are in your project path.")
    # Define placeholder functions/classes if imports fail, to allow UI to run
    # but processing will fail later. Or exit gracefully.
    # For now, let's raise the error so the user knows immediately.
    raise

# --- Global/Initial Settings (can be moved to config later if needed) ---
# image_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M/M1005'  # This will be selected by user
# output_folder = 'output/detection_results_video'  # Not saving video in live display
# video_filename = 'output_video.mp4' # Not saving video in live display
yaml_path = 'configs/DE_NET.yaml'  # Your config YAML file path
max_size = 1024  # Image resize max side length, consistent with your model
# conf_threshold = 0.25  # Initial threshold, controlled by slider
# iou_threshold = 0.4 # Initial threshold, controlled by slider (assuming your model uses this internally or in post-processing)
# output_fps = 30 # Display FPS, not saving video FPS

# ---- Image Preprocessing Transform (defined once) ----
transform = transforms.Compose([transforms.ToTensor()])

# ---- Utility Functions (from your script) ----
def preprocess_image(image_pil, max_size):
    """Preprocesses PIL image: to tensor, resize, add batch dim."""
    orig_w, orig_h = image_pil.size
    image_tensor = transform(image_pil)

    # Calculate resize scale and new dimensions
    r = min(1.0, max_size / float(max(orig_w, orig_h)))
    # Ensure dimensions are multiples of 32, common for many models
    new_h = max(32, int(math.floor(orig_h * r / 32) * 32))
    new_w = max(32, int(math.floor(orig_w * r / 32) * 32))

    image_resized = F.resize(image_tensor, (new_h, new_w))
    # Add batch dimension (B, C, H, W)
    input_tensor = image_resized.unsqueeze(0)

    # Return input tensor and original/new dimensions for scaling boxes
    return input_tensor, (orig_w, orig_h), (new_w, new_h)

def scale_boxes_to_original(boxes, orig_dims, new_dims):
    """Scales predicted boxes from resized image coords back to original image coords."""
    orig_w, orig_h = orig_dims
    new_w, new_h = new_dims

    # Calculate scaling factors
    scale_w = orig_w / new_w
    scale_h = orig_h / new_h

    scaled_boxes = []
    for box in boxes:
        # box format: [x1, y1, x2, y2, conf, ...]
        if len(box) < 4:
            continue
        # Scale coordinates
        x1, y1, x2, y2 = box[:4]
        scaled_x1 = x1 * scale_w
        scaled_y1 = y1 * scale_h
        scaled_x2 = x2 * scale_w
        scaled_y2 = y2 * scale_h

        # Append scaled coordinates and remaining elements (conf, class_id, etc.)
        scaled_boxes.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2] + list(box[4:]))

    return scaled_boxes

# ---- 1. 模型加载线程 ----
class ModelLoaderThread(QThread):
    model_loaded = pyqtSignal(object, str) # Signal emitted when model is loaded (model_obj, device_str)
    loading_status = pyqtSignal(str) # Signal to update status text
    loading_error = pyqtSignal(str) # Signal for errors during loading

    def run(self):
        self.loading_status.emit("Loading configuration...")
        try:
            cfg = load_config(yaml_path)
            device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
            self.loading_status.emit(f"Using device: {device}. Loading model...")

            # --- IMPORTANT: Replace with your actual model loading and warm-up code ---
            model = create_model(cfg)  # Create model instance
            model.load_model()  # Load weights
            model.to(device)
            model.eval()  # Set model to evaluation mode
            print("Model loaded. Performing warm-up...")
            self.loading_status.emit("Model loaded. Performing warm-up...")

            # Simulate warm-up by processing a dummy tensor
            try:
                dummy_input = torch.randn(1, 3, max_size, max_size).to(device) # Adjust size if needed
                with torch.no_grad():
                     # Assuming dehaze and predict are the main parts used
                     _ = model.dehaze(dummy_input)
                     _ = model.predict(dummy_input)
                print("Warm-up complete.")
                self.loading_status.emit("Model & Warm-up complete.")
            except Exception as warm_up_e:
                 print(f"Warm-up failed: {warm_up_e}")
                 self.loading_status.emit(f"Model loaded, but warm-up failed: {warm_up_e}")
                 # Decide if warm-up failure is critical - for now, proceed but warn

            # --- End of placeholder ---

            self.model_loaded.emit(model, str(device)) # Emit the loaded model object and device
            print("Model loader thread finished.")

        except Exception as e:
            error_msg = f"Error loading model or config: {e}"
            print(error_msg)
            self.loading_error.emit(error_msg)
            self.model_loaded.emit(None, "cpu") # Indicate failure


# ---- 2. 视频处理线程 ----
class VideoProcessorThread(QThread):
    frame_processed = pyqtSignal(np.ndarray) # Signal emitted with processed frame (numpy array BGR)
    fps_updated = pyqtSignal(float) # Signal emitted with current FPS
    processing_status = pyqtSignal(str) # Signal to update status text
    finished = pyqtSignal() # Signal emitted when processing is finished
    processing_error = pyqtSignal(str) # Signal for errors during processing

    def __init__(self, model, device, frame_files, max_size=1024, initial_conf=0.5, initial_iou=0.4):
        super().__init__()
        self.model = model
        self.device = device
        self.frame_files = frame_files
        self.max_size = max_size

        # Use a mutex for thread-safe access to thresholds if they can change mid-frame processing
        # In this design, the sliders update instance variables directly, which is generally safe
        # between frames, but a mutex is safer if threshold could be read *during* inference.
        # For simplicity here, we'll rely on the update happening between frame loops.
        self._confidence_threshold = initial_conf
        self._iou_threshold = initial_iou # Assuming your model/post-processing uses this
        self._isRunning = True
        self._mutex = QMutex() # Mutex for potential future needs

        # Ensure model is in eval mode and on the correct device (should be already from loader)
        self.model.to(self.device)
        self.model.eval()


    def run(self):
        print("Starting video processing thread...")
        self.processing_status.emit("Processing video frames...")
        frame_count = len(self.frame_files)
        start_time = time.time()
        processed_frames = 0

        # Use torch.no_grad() for inference
        with torch.no_grad():
            for i, frame_path in enumerate(self.frame_files):
                if not self._isRunning:
                    print("Processing stopped by user request.")
                    break # Stop processing if requested

                try:
                    # --- Frame Reading ---
                    image_pil = Image.open(frame_path).convert("RGB")

                    # --- Preprocessing ---
                    # Move tensor to device inside the loop for robustness, though model.to(device) is done
                    input_tensor, orig_dims, new_dims = preprocess_image(image_pil, self.max_size)
                    input_tensor = input_tensor.to(self.device) # Ensure tensor is on the correct device

                    # --- Inference ---
                    # Assuming predict uses the confidence and iou thresholds internally
                    # If your model.predict method doesn't take conf/iou, you'll need to
                    # modify it or filter the results *after* predict based on thresholds.
                    # Let's assume predict *can* take thresholds or you filter results below.

                    # Get dehazed image (optional, but used in your script for drawing)
                    dehazed_tensor = self.model.dehaze(input_tensor)
                    # Get predictions
                    # IMPORTANT: Adjust this call based on your model's predict method signature
                    # If predict takes thresholds:
                    # predictions = self.model.predict(input_tensor, conf_thres=self._confidence_threshold, iou_thres=self._iou_threshold)
                    # If predict returns raw results and you filter:
                    raw_predictions = self.model.predict(input_tensor) # Assuming this returns raw results
                    # Filter results based on current thresholds *here* if needed
                    predictions = [det for det in raw_predictions if det[4] >= self._confidence_threshold] # Filter by confidence
                    # Note: IoU filtering is typically part of NMS (Non-Maximum Suppression),
                    # which is often done *inside* the model's predict or post-processing.
                    # Ensure your model's predict or a subsequent step applies NMS with the iou_threshold.
                    # If not, you'd need to implement NMS here or modify your model class.


                    # --- Postprocessing ---
                    scaled_predictions = scale_boxes_to_original(predictions, orig_dims, new_dims)

                    # Convert dehazed tensor to OpenCV format (NumPy BGR uint8) for drawing
                    # Shape (H, W, C), range [0, 255], type uint8
                    # Permute from (C, H, W) to (H, W, C)
                    # Ensure tensor is on CPU before converting to numpy
                    dehazed_np = (dehazed_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    # Convert RGB to BGR for OpenCV drawing functions
                    img_to_draw = cv2.cvtColor(dehazed_np, cv2.COLOR_RGB2BGR)

                    # Draw bounding boxes and labels on img_to_draw
                    for det in scaled_predictions:
                        if len(det) < 5: continue # Skip malformed detections
                        x1, y1, x2, y2, conf = det[:5]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Draw rectangle
                        cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box

                        # Add text (confidence)
                        label = f"{conf:.2f}"
                        text_x, text_y = x1, y1 - 5
                        text_y = max(text_y, 15) # Avoid drawing off top edge
                        cv2.putText(img_to_draw, label, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) # Yellow text


                    # --- Emit processed frame and update FPS ---
                    self.frame_processed.emit(img_to_draw)
                    processed_frames += 1

                    # Calculate and update FPS periodically
                    # Use a small delay to control playback speed if files are processed too fast
                    # time.sleep(0.01) # Adjust as needed

                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    if elapsed_time > 0:
                         # Calculate FPS over the last few frames for smoother display
                         # Or simply use processed_frames / elapsed_time
                         fps = processed_frames / elapsed_time
                         self.fps_updated.emit(fps)

                except Exception as e:
                    error_msg = f"Error processing frame {frame_path}: {e}"
                    print(error_msg)
                    self.processing_error.emit(error_msg)
                    # Decide whether to continue or stop on error
                    # continue # To skip this frame and continue
                    # break # To stop processing entirely on first error

        # --- Processing finished ---
        elapsed_time = time.time() - start_time
        final_fps = processed_frames / elapsed_time if elapsed_time > 0 else 0
        self.fps_updated.emit(final_fps) # Update final FPS
        self.processing_status.emit(f"Processing finished. Processed {processed_frames} frames.")
        print("Video processing thread finished.")
        self.finished.emit() # Signal that processing is done

    def stop(self):
        """Safely stop the processing thread."""
        self._isRunning = False
        # self.wait() # Wait can cause deadlock if called from a slot connected to this thread
        # Instead, just set the flag. The loop check will handle exiting.
        print("Stop requested for video processor thread.")

    # Method to update thresholds from the main thread (thread-safe way)
    # @pyqtSlot(int) # This decorator is for slots, not regular methods called via signals
    def update_confidence_threshold(self, value):
        """Update the confidence threshold based on slider value (0-100)."""
        # Use mutex if thresholds could be read mid-inference, otherwise direct set is fine
        # self._mutex.lock()
        self._confidence_threshold = value / 100.0
        # self._mutex.unlock()
        print(f"Worker updated confidence threshold to: {self._confidence_threshold:.2f}")

    # @pyqtSlot(int) # This decorator is for slots, not regular methods called via signals
    def update_iou_threshold(self, value):
        """Update the IoU threshold based on slider value (0-100)."""
        # self._mutex.lock()
        self._iou_threshold = value / 100.0
        # self._mutex.unlock()
        print(f"Worker updated IoU threshold to: {self._iou_threshold:.2f}")


# --- 3. 主窗口类 ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = None
        self.device = "cpu" # Store device string
        self.video_processor_thread = None
        self.model_loader_thread = None
        self.current_frame_files = []
        self.max_image_size = max_size # Use the global max_size

        self.setWindowTitle("Drone Detection & Tracking (Foggy)")
        self.setGeometry(100, 100, 1200, 900) # Initial window size

        # Central Widget and Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top Area: FPS Label ---
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.fps_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        main_layout.addWidget(self.fps_label)

        # --- Middle Area: Video Display ---
        self.video_display_label = QLabel("Loading model...") # Placeholder text
        self.video_display_label.setAlignment(Qt.AlignCenter)
        self.video_display_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.video_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow it to grow
        self.video_display_label.setScaledContents(True) # Scale pixmap to label size
        main_layout.addWidget(self.video_display_label)

        # --- Bottom Area: Controls ---
        control_layout = QVBoxLayout()

        # Folder Selection
        folder_select_layout = QHBoxLayout()
        self.select_folder_button = QPushButton("Select Video Frame Folder")
        self.select_folder_button.clicked.connect(self.select_folder)
        self.select_folder_button.setEnabled(False) # Disable until model loaded
        folder_select_layout.addWidget(self.select_folder_button)
        self.folder_path_label = QLabel("No folder selected")
        folder_select_layout.addWidget(self.folder_path_label)
        control_layout.addLayout(folder_select_layout)

        # Threshold Sliders
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100) # Represents 0.0 to 1.0
        self.confidence_slider.setValue(int(conf_threshold * 100)) # Default from script or 0.25
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        self.confidence_slider.setEnabled(False) # Disable until model loaded
        self.confidence_value_label = QLabel(f"{self.confidence_slider.value()/100.0:.2f}")
        self.confidence_slider.valueChanged.connect(self.update_confidence_value_display)
        # Connection to worker will be made when worker is created
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_value_label)
        control_layout.addLayout(confidence_layout)

        iou_layout = QHBoxLayout()
        iou_layout.addWidget(QLabel("IoU Threshold:"))
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100) # Represents 0.0 to 1.0
        self.iou_slider.setValue(int(iou_threshold * 100)) # Default from script or 0.40
        self.iou_slider.setTickPosition(QSlider.TicksBelow)
        self.iou_slider.setTickInterval(10)
        self.iou_slider.setEnabled(False) # Disable until model loaded
        self.iou_value_label = QLabel(f"{self.iou_slider.value()/100.0:.2f}")
        self.iou_slider.valueChanged.connect(self.update_iou_value_display)
        # Connection to worker will be made when worker is created
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_value_label)
        control_layout.addLayout(iou_layout)

        main_layout.addLayout(control_layout)

        # --- Status Bar ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Initializing...")

        # --- Start Model Loading ---
        self.start_model_loader()

    def start_model_loader(self):
        """Starts the model loading thread."""
        if self.model_loader_thread is not None and self.model_loader_thread.isRunning():
             print("Model loader is already running.")
             return

        self.model_loader_thread = ModelLoaderThread()
        self.model_loader_thread.model_loaded.connect(self.on_model_loaded)
        self.model_loader_thread.loading_status.connect(self.statusBar.showMessage)
        self.model_loader_thread.loading_error.connect(self.on_loading_error)
        self.model_loader_thread.start()

    @pyqtSlot(object, str)
    def on_model_loaded(self, model, device_str):
        """Handles model object and device string received from loader thread."""
        if model is not None:
            self.model = model
            self.device = device_str
            self.statusBar.showMessage(f"Model is ready on {self.device}. Select a folder.")
            self.select_folder_button.setEnabled(True)
            self.confidence_slider.setEnabled(True)
            self.iou_slider.setEnabled(True)
            self.video_display_label.setText(f"Model loaded on {self.device}.\nSelect a folder to start.")
            print(f"Model loaded successfully on {self.device}.")
        else:
            # Error handled by on_loading_error, just ensure controls stay disabled
            self.statusBar.showMessage("Failed to load model. Check console for details.")
            self.video_display_label.setText("Model loading failed.")
            self.select_folder_button.setEnabled(False)
            self.confidence_slider.setEnabled(False)
            self.iou_slider.setEnabled(False)
            print("Model loading failed.")

    @pyqtSlot(str)
    def on_loading_error(self, message):
        """Handles errors reported by the model loader thread."""
        self.statusBar.showMessage(f"Loading Error: {message}")
        self.video_display_label.setText(f"Loading Error: {message}")
        self.select_folder_button.setEnabled(False)
        self.confidence_slider.setEnabled(False)
        self.iou_slider.setEnabled(False)


    def select_folder(self):
        """Opens a file dialog to select the folder of video frames."""
        if self.model is None:
            self.statusBar.showMessage("Model not loaded yet.")
            print("Attempted to select folder before model loaded.")
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Select Video Frame Folder")
        if folder_path:
            self.folder_path_label.setText(f"Folder: {os.path.basename(folder_path)}")
            self.statusBar.showMessage(f"Selected folder: {folder_path}")

            # Find image files (adjust patterns as needed)
            image_patterns = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
            all_files = []
            for pattern in image_patterns:
                all_files.extend(glob.glob(os.path.join(folder_path, pattern)))

            # Sort files naturally (e.g., frame_1.jpg, frame_10.jpg, frame_2.jpg -> frame_1, frame_2, frame_10)
            # A simple sort by name works well if names are zero-padded (e.g., frame_0001.jpg)
            # For more complex sorting (like frame_1, frame_2, ..., frame_10), you might need natsort
            # pip install natsort -> from natsort import natsorted -> self.current_frame_files = natsorted(all_files)
            self.current_frame_files = sorted(all_files)

            if not self.current_frame_files:
                self.statusBar.showMessage("No image files found in the selected folder.")
                self.video_display_label.setText("No image files found.\nSelect another folder.")
                # Stop any running processor if no files found
                if self.video_processor_thread and self.video_processor_thread.isRunning():
                    self.video_processor_thread.stop()
                    # Don't wait here, let the thread finish naturally if it's mid-loop
                self.video_processor_thread = None # Ensure thread reference is cleared
                print("No image files found, processing stopped/not started.")
                return

            # Stop existing thread if running
            if self.video_processor_thread is not None and self.video_processor_thread.isRunning():
                print("Stopping existing video processor thread...")
                self.video_processor_thread.stop()
                # Wait for the thread to signal finished or exit its loop
                # It's generally safer to wait in closeEvent or rely on the thread's internal loop check
                # self.video_processor_thread.wait() # Avoid waiting in a slot if possible
                # Disconnect signals from the old thread to prevent issues
                try:
                    self.video_processor_thread.frame_processed.disconnect(self.update_image_display)
                    self.video_processor_thread.fps_updated.disconnect(self.update_fps_display)
                    self.video_processor_thread.processing_status.disconnect(self.statusBar.showMessage)
                    self.video_processor_thread.finished.disconnect(self.on_processing_finished)
                    self.video_processor_thread.processing_error.disconnect(self.on_processing_error)
                    # Disconnect slider signals from the old worker
                    self.confidence_slider.valueChanged.disconnect(self.video_processor_thread.update_confidence_threshold)
                    self.iou_slider.valueChanged.disconnect(self.video_processor_thread.update_iou_threshold)
                except TypeError: # Disconnect might raise TypeError if already disconnected
                    pass
                self.video_processor_thread = None # Clear reference immediately after stopping

            # Start new processing thread
            current_confidence = self.confidence_slider.value() / 100.0
            current_iou = self.iou_slider.value() / 100.0

            self.video_processor_thread = VideoProcessorThread(
                self.model,
                self.device,
                self.current_frame_files,
                max_size=self.max_image_size,
                initial_conf=current_confidence,
                initial_iou=current_iou
            )
            # Connect signals from the new worker thread to MainWindow slots
            self.video_processor_thread.frame_processed.connect(self.update_image_display)
            self.video_processor_thread.fps_updated.connect(self.update_fps_display)
            self.video_processor_thread.processing_status.connect(self.statusBar.showMessage)
            self.video_processor_thread.finished.connect(self.on_processing_finished)
            self.video_processor_thread.processing_error.connect(self.on_processing_error)

            # Connect slider value changes to the worker thread's update methods
            # Note: These are direct connections from a signal to a slot/method in *another* thread.
            # PyQt handles the necessary queuing.
            self.confidence_slider.valueChanged.connect(self.video_processor_thread.update_confidence_threshold)
            self.iou_slider.valueChanged.connect(self.video_processor_thread.update_iou_threshold)


            self.video_processor_thread.start()
            self.statusBar.showMessage("Processing video frames...")
            self.video_display_label.setText("Processing...")
            print("New video processor thread started.")

    @pyqtSlot(np.ndarray)
    def update_image_display(self, cv_img):
        """Updates the QLabel with a new frame from OpenCV (BGR numpy array)."""
        if cv_img is None:
            return

        # Convert OpenCV image (BGR) to QImage (RGB)
        # Ensure the image is contiguous for QImage constructor
        cv_img = np.ascontiguousarray(cv_img)
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        # QImage needs RGB, OpenCV is BGR, so swap channels
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Set QImage to QLabel as QPixmap
        pixmap = QPixmap.fromImage(q_img)
        # The label's setScaledContents(True) will handle scaling
        self.video_display_label.setPixmap(pixmap)
        # self.video_display_label.setAlignment(Qt.AlignCenter) # Keep image centered

    @pyqtSlot(float)
    def update_fps_display(self, fps):
        """Updates the FPS label."""
        self.fps_label.setText(f"FPS: {fps:.2f}")

    @pyqtSlot(int)
    def update_confidence_value_display(self, value):
        """Updates the text label next to the confidence slider."""
        self.confidence_value_label.setText(f"{value/100.0:.2f}")

    @pyqtSlot(int)
    def update_iou_value_display(self, value):
        """Updates the text label next to the IoU slider."""
        self.iou_value_label.setText(f"{value/100.0:.2f}")

    @pyqtSlot()
    def on_processing_finished(self):
        """Handles the signal when the video processing thread finishes."""
        self.statusBar.showMessage("Processing finished.")
        # Optionally reset UI elements or prepare for next folder selection
        print("Processing finished slot triggered.")
        # Disconnect signals from the finished thread
        if self.video_processor_thread:
             try:
                self.video_processor_thread.frame_processed.disconnect(self.update_image_display)
                self.video_processor_thread.fps_updated.disconnect(self.update_fps_display)
                self.video_processor_thread.processing_status.disconnect(self.statusBar.showMessage)
                self.video_processor_thread.finished.disconnect(self.on_processing_finished)
                self.video_processor_thread.processing_error.disconnect(self.on_processing_error)
                # Disconnect slider signals from the finished worker
                self.confidence_slider.valueChanged.disconnect(self.video_processor_thread.update_confidence_threshold)
                self.iou_slider.valueChanged.disconnect(self.video_processor_thread.update_iou_threshold)
             except TypeError:
                pass
             self.video_processor_thread = None # Clear the reference

    @pyqtSlot(str)
    def on_processing_error(self, message):
        """Handles errors reported by the video processing thread."""
        self.statusBar.showMessage(f"Processing Error: {message}")
        print(f"Processing Error: {message}")
        # Consider stopping the thread on error if it doesn't stop itself
        if self.video_processor_thread and self.video_processor_thread.isRunning():
             self.video_processor_thread.stop()
             # The thread's finished signal should handle cleanup

    def closeEvent(self, event):
        """Ensures threads are stopped before closing the application."""
        print("Closing application. Stopping threads...")
        # It's crucial to stop threads gracefully before the application exits

        if self.model_loader_thread is not None and self.model_loader_thread.isRunning():
            print("Stopping model loader thread...")
            self.model_loader_thread.quit() # Request the thread to exit its event loop (if any)
            self.model_loader_thread.wait(5000) # Wait up to 5 seconds

        if self.video_processor_thread is not None and self.video_processor_thread.isRunning():
            print("Stopping video processor thread...")
            self.video_processor_thread.stop() # Set the internal flag
            self.video_processor_thread.wait(5000) # Wait up to 5 seconds

        if (self.model_loader_thread is not None and self.model_loader_thread.isRunning()) or \
           (self.video_processor_thread is not None and self.video_processor_thread.isRunning()):
            print("Warning: One or more threads did not stop cleanly.")
            # Force exit might be needed in extreme cases, but try to avoid
            # sys.exit(app.exec_()) # This will be called anyway
            pass # Let the event proceed

        print("All threads stopped or timed out. Accepting close event.")
        event.accept() # Accept the window close event


# --- Main execution ---
if __name__ == "__main__":
    # Define default thresholds if they weren't defined globally in the script
    try:
        conf_threshold # Check if already defined
    except NameError:
        conf_threshold = 0.25 # Default confidence threshold

    try:
        iou_threshold # Check if already defined
    except NameError:
        iou_threshold = 0.40 # Default IoU threshold

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
