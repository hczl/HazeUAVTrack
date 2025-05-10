import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import time
import math
import torch
import numpy as np
import cv2
from PIL import Image, ImageTk
import torchvision.transforms.functional as F
from torchvision import transforms
from tqdm import tqdm # Using tqdm for console output in the worker thread

try:
    from utils.config import load_config
    from utils.create import create_model
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    print("Please ensure 'utils' folder with config.py and create.py is in your project path.")
    # Exit or handle the error appropriately if utilities are essential
    raise
os.environ['TORCH_HOME'] = './.torch'
# ---- Global/Shared Variables for Thread Communication ----
current_frame_data = None # Holds (PhotoImage, status_text) or similar
processing_running = False # Flag to indicate if the worker thread is active
stop_event = threading.Event() # Event to signal the worker thread to stop

# ---- Model and Processing Globals (Loaded by worker) ----
model = None # Global variable to hold the model instance
device = None
cfg = None
transform = transforms.Compose([transforms.ToTensor()])
image_files = []
current_frame_index = -1 # -1 means not started or finished
max_size = 1024 # Default max image size

# ---- Image Processing and Detection Functions (Adapted from your script) ----
def preprocess_image(image_pil, max_size):
    """Preprocesses PIL image for model input: to tensor, resize, add batch dim."""
    orig_w, orig_h = image_pil.size
    image_tensor = transform(image_pil)

    # Calculate resize scale and new dimensions for model input
    # F.resize with (h, w) tuple stretches/squashes, it doesn't maintain aspect ratio or pad
    r = min(1.0, max_size / float(max(orig_w, orig_h)))
    # Calculate new dimensions ensuring they are multiples of 32 (common requirement for models)
    new_h = max(32, int(math.floor(orig_h * r / 32) * 32))
    new_w = max(32, int(math.floor(orig_w * r / 32) * 32))

    image_resized_tensor = F.resize(image_tensor, (new_h, new_w))
    # Assuming model expects float tensor in [0, 1]
    input_tensor = image_resized_tensor.unsqueeze(0).to(device)

    # Return input tensor and original/resized dimensions
    return input_tensor, (orig_w, orig_h), (new_w, new_h)

def scale_boxes_to_original(boxes, orig_dims, new_dims):
    """Scales predicted boxes from resized image coords back to original image coords."""
    orig_w, orig_h = orig_dims
    new_w, new_h = new_dims

    # Calculate scaling factors based on simple stretch/squash resize
    # Ensure no division by zero if new_w or new_h are zero (shouldn't happen if max_size > 0)
    scale_w = orig_w / new_w if new_w > 0 else 1.0
    scale_h = orig_h / new_h if new_h > 0 else 1.0

    scaled_boxes = []
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        # Apply scaling
        x1 *= scale_w
        y1 *= scale_h
        x2 *= scale_w
        y2 *= scale_h
        scaled_boxes.append([x1, y1, x2, y2] + list(box[4:])) # Keep confidence, class, etc.

    return scaled_boxes


def draw_boxes(image_np_original_size, detections):
    """Draws bounding boxes and labels on the OpenCV image (BGR format) at original size."""
    # image_np_original_size: OpenCV image in BGR format, uint8, original dimensions
    # detections: List of detections, coordinates MUST be relative to original dimensions.
    # Assumes detections are *already filtered* by confidence/IoU by the model itself.
    img_to_draw = image_np_original_size.copy() # Draw on a copy

    # We no longer filter by confidence here, assuming the model output is already filtered
    filtered_detections = detections # Use all detections provided
    for det in filtered_detections:
        # Ensure the detection has at least 5 elements (x1, y1, x2, y2, conf)
        if len(det) < 5:
            continue # Skip malformed detections

        x1, y1, x2, y2, conf = det[:5] # Use only the first 5 elements
        # Ensure coordinates are integers for OpenCV drawing
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw rectangle (OpenCV uses BGR color format)
        # Green color (0, 255, 0), thickness 2
        cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text annotation (confidence)
        text_x, text_y = x1, y1 - 10 # Offset slightly above the box
        text_y = max(text_y, 15) # Ensure text is not off-screen at the top

        label = f"{conf:.2f}"

        # Draw text (OpenCV uses BGR color format)
        # Yellow color (0, 255, 255), font scale 0.5, thickness 2
        cv2.putText(img_to_draw, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return img_to_draw


# ---- Worker Thread Function ----
def process_video_frames(yaml_path, image_folder, conf_var, iou_var, status_var, video_label_size):
    """
    This function runs in a separate thread to perform video processing.
    It updates shared variables that the UI thread reads.
    It also sets model properties based on slider values.
    """
    global model, device, cfg, image_files, current_frame_index, processing_running, current_frame_data, max_size

    processing_running = True
    current_frame_index = -1 # Reset frame index

    # --- Load Config and Model ---
    try:
        status_var.set("Status: Loading config...")
        cfg = load_config(yaml_path)
        max_size = cfg.get('max_size', 1024) # Get max_size from config if available

        status_var.set("Status: Creating model...")
        model = create_model(cfg)

        status_var.set("Status: Loading model weights...")
        model.load_model() # Load weights

        device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # Set model to evaluation mode

        # --- Set Initial Model Thresholds from Config ---
        initial_conf = cfg.get('detector_conf_thresh', 0.25)
        initial_iou = cfg.get('detector_iou_thresh', 0.45)
        try:
            # Attempt to set model attributes directly
            if hasattr(model, 'conf_thresh'):
                model.conf_thresh = initial_conf
                conf_var.set(initial_conf) # Also set UI slider to match initial config
            else:
                 print("Warning: Model object does not have a 'conf_thresh' attribute.")
                 print("Confidence slider will control display filtering only (if draw_boxes was filtering).")


            if hasattr(model, 'iou_thresh'):
                model.iou_thresh = initial_iou
                iou_var.set(initial_iou) # Also set UI slider to match initial config
            else:
                 print("Warning: Model object does not have an 'iou_thresh' attribute.")
                 print("IoU slider value will only be displayed.")

            status_var.set(f"Status: Model loaded on {device}. Initial thresholds: Conf={initial_conf:.2f}, IoU={initial_iou:.2f}. Warming up...")

        except Exception as e:
             print(f"Error setting initial model thresholds: {e}")
             status_var.set(f"Status: Model loaded on {device}. Error setting initial thresholds ({e}). Warming up...")


        # --- Get Image Files ---
        image_files = sorted([
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not image_files:
            status_var.set(f"Error: No image files found in {image_folder}")
            processing_running = False
            # Clean up model if loading failed
            if model is not None:
                del model
                model = None
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()
            return

    except Exception as e:
        status_var.set(f"Error during model loading: {e}")
        model = None # Ensure model is None if loading failed
        processing_running = False
        return

    status_var.set("Status: Model ready. Starting inference...")
    time.sleep(1) # Give UI a moment to update status

    start_time = time.time()
    frame_count = 0

    # --- Main Processing Loop ---
    for i, image_path in enumerate(tqdm(image_files, desc="Processing Frames")):
        if stop_event.is_set():
            status_var.set("Status: Processing stopped.")
            break # Stop processing if the stop event is set

        current_frame_index = i
        frame_start_time = time.time()

        # --- Get current thresholds from UI sliders ---
        # Read the values from the shared DoubleVar objects *before* prediction
        current_conf_thresh = conf_var.get()
        current_iou_thresh = iou_var.get()

        # --- Set Model Thresholds for the current frame ---
        # This is where we update the model's internal state
        try:
            if model is not None: # Ensure model was loaded successfully
                if hasattr(model, 'conf_thresh'):
                    model.conf_thresh = current_conf_thresh
                if hasattr(model, 'iou_thresh'):
                    model.iou_thresh = current_iou_thresh
        except Exception as e:
            # This warning can be noisy, perhaps log it or handle differently
            # print(f"Warning: Could not set model thresholds for frame {i}: {e}")
            pass # Suppress frequent warnings


        try:
            # Load original image for drawing later
            image_pil_original = Image.open(image_path).convert("RGB")
            orig_w, orig_h = image_pil_original.size
            # Convert original PIL image to OpenCV BGR format for drawing
            image_np_original_bgr = cv2.cvtColor(np.array(image_pil_original), cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"Worker: Skipping file {image_path}: Cannot open or process image ({e})")
            continue # Skip to the next image

        # 1. Preprocess image (for model input - this still needs resizing)
        input_tensor, orig_dims, new_dims = preprocess_image(image_pil_original, max_size)

        # 2. Dehaze and Predict
        with torch.no_grad():
            # Get dehazed image (this is in the resized dimensions from model output)
            # We will resize it back to original size for drawing.
            if model is not None: # Ensure model is available
                dehazed_tensor = model.dehaze(input_tensor)
                # Convert dehazed tensor to OpenCV format (NumPy BGR uint8) - still resized
                dehazed_np_resized = (dehazed_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                dehazed_np_resized = cv2.cvtColor(dehazed_np_resized, cv2.COLOR_RGB2BGR)

                # Get predictions. The model is expected to use the internal
                # conf_thresh and iou_thresh set above to filter predictions.
                predictions = model.predict(input_tensor) # These coordinates are relative to new_dims
            else: # Model failed to load, use placeholder
                 dehazed_np_resized = cv2.resize(image_np_original_bgr, (new_dims[0], new_dims[1]), interpolation=cv2.INTER_LINEAR)
                 predictions = [] # No predictions if model failed


        # 3. Scale predicted boxes from resized coords back to original coords
        # These predictions should already be filtered by the model's internal thresholds
        scaled_predictions = scale_boxes_to_original(predictions, orig_dims, new_dims)

        # 4. Resize the dehazed image back to original size for drawing
        dehazed_np_original_size = cv2.resize(dehazed_np_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


        # 5. Draw results on the original-sized dehazed image.
        # No confidence filtering is done here, assuming the model did it.
        img_to_draw = draw_boxes(dehazed_np_original_size, scaled_predictions)

        # 6. Calculate FPS and prepare status text
        frame_end_time = time.time()
        frame_time = frame_end_time - frame_start_time
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        status_text = f"FPS: {fps:.2f} | Conf: {current_conf_thresh:.2f} | IoU: {current_iou_thresh:.2f} | Frame: {i+1}/{len(image_files)}"

        # 7. Prepare the final drawn-on image (which is original size) for Tkinter display
        # Convert BGR OpenCV image to RGB PIL Image
        img_rgb_original_size = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
        img_pil_original_size = Image.fromarray(img_rgb_original_size)

        # Resize image to fit video label size while maintaining aspect ratio
        if video_label_size and video_label_size[0] > 0 and video_label_size[1] > 0:
             img_pil_display = img_pil_original_size.copy() # Work on a copy
             img_pil_display.thumbnail(video_label_size, Image.Resampling.LANCZOS) # Use thumbnail for aspect-preserving resize
        else:
             img_pil_display = img_pil_original_size # Use original size if label size is zero or invalid

        img_tk = ImageTk.PhotoImage(img_pil_display)

        # Update shared variables.
        global current_frame_data
        current_frame_data = (img_tk, status_text)

        # Small sleep
        # time.sleep(0.005) # Adjust as needed

    # --- Processing Finished ---
    status_var.set("Status: Processing complete.")
    processing_running = False
    current_frame_index = len(image_files) # Indicate finished

    # Clean up model
    if model is not None:
        del model
        model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- Tkinter UI Class ----
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Video Inference UI")
        self.geometry("800x700") # Initial window size

        # Variables for paths with default values
        # !!! Set your desired default paths here !!!
        self.yaml_path_var = tk.StringVar(value="configs/DRIFT_NET.yaml") # Example default
        self.video_folder_var = tk.StringVar(value="data/UAV-M/MiDaS_Deep_UAV-benchmark-M_fog_050/M1005") # Example default
        # !!! ---------------------------------- !!!


        # Variables for sliders
        self.conf_var = tk.DoubleVar(value=0.25) # Default confidence (will be updated from config)
        self.iou_var = tk.DoubleVar(value=0.45)  # Default IoU (will be updated from config)

        # Variable for status text
        self.status_var = tk.StringVar(value="Status: Waiting for input...")

        # Processing thread handle
        self.processing_thread = None

        # Keep a reference to the current PhotoImage to prevent garbage collection
        self.current_tk_image = None

        # UI Setup
        self._setup_input_ui()
        self._setup_video_ui()

        # Hide video UI initially
        self.video_frame.pack_forget()

        # Handle window closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Schedule periodic UI updates
        self._update_frame_display_scheduled = None
        # We don't start the update loop until processing begins

        # Set initial button states
        self._set_input_ui_state(True)
        self._set_video_ui_state(False) # Hide video UI buttons initially


    def _setup_input_ui(self):
        self.input_frame = ttk.Frame(self, padding="10")
        self.input_frame.pack(fill=tk.BOTH, expand=True)

        # YAML Path Input
        ttk.Label(self.input_frame, text="Model YAML:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.yaml_entry = ttk.Entry(self.input_frame, textvariable=self.yaml_path_var, width=50)
        self.yaml_entry.grid(row=0, column=1, pady=5, padx=5)
        self.yaml_browse_btn = ttk.Button(self.input_frame, text="Browse", command=self._browse_yaml)
        self.yaml_browse_btn.grid(row=0, column=2, pady=5, padx=5)

        # Video Folder Input
        ttk.Label(self.input_frame, text="Video Folder:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.video_folder_entry = ttk.Entry(self.input_frame, textvariable=self.video_folder_var, width=50)
        self.video_folder_entry.grid(row=1, column=1, pady=5, padx=5)
        self.video_folder_browse_btn = ttk.Button(self.input_frame, text="Browse", command=self._browse_folder)
        self.video_folder_browse_btn.grid(row=1, column=2, pady=5, padx=5)

        # Start Button
        self.start_button = ttk.Button(self.input_frame, text="Start Processing", command=self._start_processing)
        self.start_button.grid(row=2, column=0, columnspan=3, pady=20)

        # Configure grid to expand
        self.input_frame.columnconfigure(1, weight=1)

    def _setup_video_ui(self):
        self.video_frame = ttk.Frame(self, padding="10")
        # Note: This frame is not packed initially

        # Status Label (FPS, Thresholds, Frame count)
        self.status_label = ttk.Label(self.video_frame, textvariable=self.status_var)
        self.status_label.pack(pady=5)

        # Video Display Area
        self.video_label = ttk.Label(self.video_frame, text="Model warming up...", anchor="center")
        self.video_label.pack(pady=10, fill=tk.BOTH, expand=True)
        self.video_label.config(compound=tk.CENTER) # Center text and image

        # Sliders Frame
        sliders_frame = ttk.Frame(self.video_frame)
        sliders_frame.pack(pady=10)

        # Confidence Slider
        ttk.Label(sliders_frame, text="Confidence Threshold:").grid(row=0, column=0, padx=5)
        self.conf_slider = ttk.Scale(sliders_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                     variable=self.conf_var, length=300)
        self.conf_slider.grid(row=0, column=1, padx=5)
        self.conf_value_label = ttk.Label(sliders_frame, textvariable=self.conf_var, width=5)
        self.conf_value_label.grid(row=0, column=2, padx=5)


        # IoU Slider
        ttk.Label(sliders_frame, text="IoU Threshold:").grid(row=1, column=0, padx=5)
        self.iou_slider = ttk.Scale(sliders_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                    variable=self.iou_var, length=300)
        self.iou_slider.grid(row=1, column=1, padx=5)
        self.iou_value_label = ttk.Label(sliders_frame, textvariable=self.iou_var, width=5)
        self.iou_value_label.grid(row=1, column=2, padx=5)

        # Stop and Re-select Button
        self.stop_button = ttk.Button(self.video_frame, text="Stop and Re-select Video", command=self._stop_and_reselect)
        self.stop_button.pack(pady=10)

    def _set_input_ui_state(self, enabled):
        """Enables or disables controls in the input frame."""
        state = "normal" if enabled else "disabled"
        self.yaml_entry.config(state=state)
        self.yaml_browse_btn.config(state=state)
        self.video_folder_entry.config(state=state)
        self.video_folder_browse_btn.config(state=state)
        self.start_button.config(state=state)

    def _set_video_ui_state(self, enabled):
        """Enables or disables controls in the video frame."""
        state = "normal" if enabled else "disabled"
        # status_label and video_label don't have state
        self.conf_slider.config(state=state)
        self.iou_slider.config(state=state)
        # Value labels don't have state
        self.stop_button.config(state=state)


    def _browse_yaml(self):
        filename = filedialog.askopenfilename(
            title="Select Model YAML File",
            filetypes=(("YAML files", "*.yaml"), ("All files", "*.*"))
        )
        if filename:
            self.yaml_path_var.set(filename)

    def _browse_folder(self):
        foldername = filedialog.askdirectory(
            title="Select Video Frames Folder"
        )
        if foldername:
            self.video_folder_var.set(foldername)

    def _start_processing(self):
        yaml_path = self.yaml_path_var.get()
        video_folder = self.video_folder_var.get()

        if not os.path.isfile(yaml_path):
            messagebox.showerror("Error", f"YAML file not found: {yaml_path}")
            return
        if not os.path.isdir(video_folder):
            messagebox.showerror("Error", f"Video folder not found: {video_folder}")
            return

        # Disable input UI and enable video UI controls
        self._set_input_ui_state(False)
        self._set_video_ui_state(True)

        # Switch UI frames
        self.input_frame.pack_forget()
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # Get the current size of the video label to resize images accordingly
        self.update_idletasks() # Process pending geometry updates
        video_label_width = self.video_label.winfo_width()
        video_label_height = self.video_label.winfo_height()
        video_label_size = (video_label_width, video_label_height)
        print(f"Video label size: {video_label_size}") # Debugging

        # Clear stop event and start the worker thread
        stop_event.clear()
        global processing_running, current_frame_index, current_frame_data
        processing_running = True
        current_frame_index = -1
        current_frame_data = None
        self.video_label.config(image='', text="Model warming up...") # Reset video display text

        # Pass slider variables and status variable to the worker thread
        self.processing_thread = threading.Thread(
            target=process_video_frames,
            args=(yaml_path, video_folder, self.conf_var, self.iou_var, self.status_var, video_label_size)
        )
        self.processing_thread.daemon = True # Allow the main thread to exit even if worker is running
        self.processing_thread.start()

        # Start the UI update loop
        self._schedule_update()

    def _stop_processing(self):
        """Signals the worker thread to stop and waits for it."""
        global processing_running
        if processing_running:
            stop_event.set() # Set the event to signal the worker

            # Wait briefly for the thread to acknowledge and stop
            if self.processing_thread and self.processing_thread.is_alive():
                 print("Waiting for processing thread to finish...")
                 # Use a reasonable timeout, don't block UI indefinitely
                 self.processing_thread.join(timeout=5)
                 if self.processing_thread.is_alive():
                      print("Warning: Processing thread did not stop within timeout.")
                      # You might want to handle this case, e.g., forcefully exit?
                      # For now, just print a warning and continue.

            processing_running = False
            self.status_var.set("Status: Stopped.")

            # Clean up model globally when stopping
            global model
            if model is not None:
                del model
                model = None
            if torch.cuda.is_available():
                 torch.cuda.empty_cache()

    def _stop_and_reselect(self):
        """Stops processing and switches back to the input UI."""
        self._stop_processing() # Stop the worker thread

        # Clear the displayed image and status text
        self.current_tk_image = None
        self.video_label.config(image='', text="Select video folder and YAML.")
        self.status_var.set("Status: Ready.")

        # Switch UI frames
        self.video_frame.pack_forget()
        self.input_frame.pack(fill=tk.BOTH, expand=True)

        # Enable input UI controls
        self._set_input_ui_state(True)
        self._set_video_ui_state(False) # Disable video UI controls


    def _schedule_update(self):
        """Schedules the next UI update check."""
        # Check for updates every 30 milliseconds (adjust for desired responsiveness vs CPU usage)
        self._update_frame_display_scheduled = self.after(30, self._update_frame_display)

    def _update_frame_display(self):
        """Checks for new frame data from the worker and updates the UI."""
        global current_frame_data, processing_running, current_frame_index, image_files

        # Check if new data is available
        if current_frame_data is not None:
            img_tk, status_text = current_frame_data
            self.current_tk_image = img_tk # Keep reference!
            self.video_label.config(image=self.current_tk_image, text="") # Update image, clear "warming up" text
            self.status_var.set(status_text) # Update status text

            # Clear the shared data after using it
            current_frame_data = None

        # Check if the worker thread is still running
        # Also check if processing is finished (index >= total frames)
        if processing_running:
            # Reschedule the next update
            self._schedule_frame_display_scheduled = self.after(30, self._update_frame_display)
        elif current_frame_index != -1 and current_frame_index >= len(image_files) - 1:
             # Processing finished
             print("Processing thread finished.")
             self._update_frame_display_scheduled = None # Cancel scheduled updates
             self.status_var.set("Status: Processing complete.")
             # Clean up model globally when finished
             global model
             if model is not None:
                 del model
                 model = None
             if torch.cuda.is_available():
                  torch.cuda.empty_cache()
             # Optionally switch back to input UI automatically on completion
             # self._stop_and_reselect() # Uncomment if you want auto-return


    def on_closing(self):
        """Handle the window closing event."""
        print("Closing application...")
        self._stop_processing() # Signal the worker to stop
        self.destroy() # Destroy the Tkinter window


# ---- Main Execution ----
if __name__ == "__main__":
    app = App()
    app.mainloop()
