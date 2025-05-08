import os
import time
import math
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import threading
import queue

from flask import Flask, render_template, request, Response, jsonify

# Assuming your utils are in a directory accessible by the app
# Make sure the directory containing utils is in Python's path,
# or utils is directly in your project root alongside app.py
try:
    from utils.config import load_config
    from utils.create import create_model # This should create the FSDT model
except ImportError as e:
    print(f"Error importing utility modules: {e}")
    print("Please ensure 'utils' directory is accessible and contains config.py and create.py")
    # Exit or handle the error appropriately
    # For this example, we'll let it potentially crash later if not fixed.
    pass


# ---- Flask App Setup ----
app = Flask(__name__)
app.secret_key = 'super secret key'

# ---- Define the base directory as the parent of the script's directory ----
# If app.py is in root/ui/, this correctly sets BASE_DIR to root/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Base directory set to: {BASE_DIR}")


# ---- Global State (Shared between threads) ----
state = {
    'image_folder': None,
    'yaml_path': None,
    'model': None,
    'device': None,
    'is_processing': False,
    'processing_thread': None,
    'frame_queue': queue.Queue(maxsize=10), # Queue to hold processed frames
    'current_fps': 0.0,
    'max_size': 1024, # Default max_size, should ideally come from config
    'output_fps': 30, # Target output FPS
    'image_files': [], # List of image file paths
    'orig_dims': None, # Original image dimensions (W, H)
    'new_dims': None, # Resized image dimensions (W, H)
    'transform': transforms.Compose([transforms.ToTensor()]) # Image transform
}

# ---- Helper Function to resolve paths relative to BASE_DIR ----
def resolve_path(user_input_path):
    """
    Resolves a path relative to the base directory and normalizes it
    using the native OS path separator.
    """
    if not user_input_path:
        return None

    # os.path.join correctly handles absolute paths in user_input_path
    joined_path = os.path.join(BASE_DIR, user_input_path)

    # Normalize the resulting path
    normalized_path = os.path.normpath(joined_path)

    print(f"Resolving '{user_input_path}' relative to '{BASE_DIR}' -> Joined: '{joined_path}' -> Normalized: '{normalized_path}'")
    return normalized_path

# ---- Helper Functions (adapted from your script) ----

def preprocess_image(image_pil, max_size=640):
    """Preprocesses PIL image: to tensor, resize, add batch dim, move to device."""
    orig_w, orig_h = image_pil.size
    image_tensor = state['transform'](image_pil)

    # Calculate resize scale and new dimensions
    r = min(1.0, max_size / float(max(orig_w, orig_h)))
    new_h = max(32, int(math.floor(orig_h * r / 32) * 32))
    new_w = max(32, int(math.floor(orig_w * r / 32) * 32))

    # Note: F.resize expects (h, w) tuple
    image_resized = F.resize(image_tensor, (new_h, new_w))
    input_tensor = image_resized.unsqueeze(0).to(state['device'])

    # Return input tensor and original/new dimensions for scaling boxes
    return input_tensor, (orig_w, orig_h), (new_w, new_h)

def scale_boxes_to_original(boxes, orig_dims, new_dims):
    """Scales predicted boxes from resized image coords back to original image coords."""

    # Check if 'boxes' is a PyTorch Tensor or NumPy array and if it's empty (no detections)
    # A tensor/array of detections is usually shape (N, features), where N is the number of detections.
    # If N is 0, there are no detections.
    has_detections = False
    if isinstance(boxes, torch.Tensor):
        # Check if it's at least 2D (N, features) and has more than 0 detections
        if boxes.ndim > 1 and boxes.shape[0] > 0:
             has_detections = True
    elif isinstance(boxes, np.ndarray):
         # Check if it's at least 2D (N, features) and has more than 0 detections
         if boxes.ndim > 1 and boxes.shape[0] > 0:
              has_detections = True
    elif isinstance(boxes, list):
         # If it's a list (less common for model output, but handle defensively)
         if boxes: # Check if the list is not empty
             has_detections = True

    # If no detections found based on the type and shape/emptiness check
    if not has_detections:
        # print("No detections found (empty tensor, numpy array, or list). Returning empty list.") # Optional debug print
        return [] # Return empty list as originally intended when no boxes are present


    # If we reach here, 'boxes' is assumed to contain detections (N > 0)
    orig_w, orig_h = orig_dims
    new_w, new_h = new_dims

    # Calculate scaling factors based on the dimensions used by the model
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    scaled_boxes = []

    # Iterate over the individual detection boxes.
    # This loop structure works for lists, tensors, and numpy arrays.
    for det in boxes:
        # 'det' here will be a single detection, potentially a tensor row, numpy array row, or list.
        # Ensure the detection has at least 4 coordinates (x1, y1, x2, y2)

        # Convert the detection item 'det' to a list for consistent processing
        det_list = None
        if isinstance(det, torch.Tensor):
             if det.numel() >= 4:
                  det_list = det.tolist() # Convert tensor row to list
        elif isinstance(det, np.ndarray):
             if det.size >= 4:
                  det_list = det.tolist() # Convert numpy row to list
        elif isinstance(det, list):
             if len(det) >= 4:
                  det_list = det # Already a list
        else:
             # Unexpected type for a detection item, skip
             print(f"Warning: Skipping detection item of unexpected type: {type(det)}")
             continue

        # If det_list was successfully created and has enough elements
        if det_list is not None and len(det_list) >= 4:
            x1, y1, x2, y2 = det_list[:4]

            # Apply scaling
            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

            # Keep the rest of the elements (conf, class, etc.)
            # Ensure we only take elements up to the original length of det_list
            scaled_box = [x1_scaled, y1_scaled, x2_scaled, y2_scaled] + list(det_list[4:])
            scaled_boxes.append(scaled_box)
        # else: # Optional: print a warning if a detection item was skipped due to insufficient elements
            # print(f"Warning: Skipping detection item with insufficient elements: {det}")


    return scaled_boxes

def draw_boxes_and_labels(image_np, boxes):
    """Draws bounding boxes and labels on the OpenCV image."""
    img_to_draw = image_np.copy()
    height, width, _ = img_to_draw.shape

    for det in boxes:
        if len(det) < 5: # Need at least x1, y1, x2, y2, conf
            continue

        x1, y1, x2, y2, conf = det[:5]

        # Ensure coordinates are integers and within image bounds
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        # Draw rectangle (OpenCV uses BGR color format)
        cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box

        # Add text annotation (confidence)
        label = f"{conf:.2f}"
        text_x, text_y = x1, y1 - 10
        text_y = max(text_y, 15) # Ensure text is not cut off at the top

        cv2.putText(img_to_draw, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2) # Yellow text

    return img_to_draw

def draw_fps(image_np, fps):
    """Draws the current FPS on the image."""
    img_to_draw = image_np.copy()
    # Add FPS to the image (e.g., top-left corner)
    cv2.putText(img_to_draw, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) # Yellow text, size 1, thickness 2
    return img_to_draw


# ---- Processing Thread Function ----

def process_video_thread():
    print("Processing thread started.")
    state['is_processing'] = True
    frame_count = 0
    start_time = time.time()

    try:
        # --- Start: Model Loading and Path Checks ---
        # Load config and model if not already loaded or if path changed
        if state['model'] is None:
            if not state['yaml_path'] or not os.path.isfile(state['yaml_path']):
                print(f"Error: YAML path not set or invalid: {state['yaml_path']}")
                state['is_processing'] = False
                # Signal end of stream
                try: state['frame_queue'].put(None)
                except queue.Full: pass
                return

            print(f"Loading config from resolved path: {state['yaml_path']}")
            try:
                cfg = load_config(state['yaml_path'])

                # --- Logic to get resolved weights path (if applicable) ---
                # This part depends on how your load_config and create_model/load_model work
                # If your model weights path is in the yaml, you'd resolve it here
                # Example (adjust based on your config structure):
                weights_relative_path = cfg.get('weights')
                resolved_weights_path = None
                if weights_relative_path:
                    resolved_weights_path = resolve_path(weights_relative_path)
                    if not os.path.isfile(resolved_weights_path):
                         print(f"Warning: Model weights file specified in config not found at resolved path: {resolved_weights_path}")
                         # Decide if this is a fatal error or if model loading can proceed without weights
                         # For now, let's assume it's needed
                         print("Error: Model weights file not found. Cannot proceed.")
                         state['is_processing'] = False
                         try: state['frame_queue'].put(None)
                         except queue.Full: pass
                         return
                    else:
                        print(f"Resolved model weights path: {resolved_weights_path}")
                        # IMPORTANT: Update the config or pass this path to your model loading
                        cfg['resolved_weights_path'] = resolved_weights_path # Example: add to config


                state['device'] = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else "cpu"))
                print(f"Using device: {state['device']}")

                # Create and load model - ENSURE THIS PART USES THE RESOLVED WEIGHTS PATH
                state['model'] = create_model(cfg) # Assuming create_model uses the config (which now has resolved_weights_path)
                # If load_model needs the path explicitly:
                # state['model'].load_model(resolved_weights_path) # Example if load_model takes path
                # If load_model reads from config:
                state['model'].load_model() # Assuming load_model uses cfg['resolved_weights_path'] or cfg['weights'] which you updated

                state['model'].to(state['device'])
                state['model'].eval()
                print("Model loaded successfully.")

                # Update max_size from config
                state['max_size'] = cfg.get('input_size', state['max_size'])
                print(f"Using max_size: {state['max_size']}")

            except Exception as e:
                 print(f"Error loading model or config: {e}")
                 import traceback
                 traceback.print_exc()
                 state['is_processing'] = False
                 # Signal end of stream
                 try: state['frame_queue'].put(None)
                 except queue.Full: pass
                 return

        # Ensure the image folder path is valid *after* resolving
        if not state['image_folder'] or not os.path.isdir(state['image_folder']):
             print(f"Error: Invalid image folder path: {state['image_folder']}")
             state['is_processing'] = False
             # Signal end of stream
             try: state['frame_queue'].put(None)
             except queue.Full: pass
             return
        # --- End: Model Loading and Path Checks ---


        # --- Start: Get image files list with more debugging (from previous response) ---
        print(f"Attempting to list files in resolved folder: {state['image_folder']}")

        try:
            dir_contents = os.listdir(state['image_folder'])
            print(f"Raw directory contents found: {dir_contents}")

            image_files_list = [
                os.path.join(state['image_folder'], f)
                for f in dir_contents
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]

            state['image_files'] = sorted(image_files_list)

            print(f"Found {len(state['image_files'])} image files matching extensions (.jpg, .jpeg, .png).")

        except Exception as e:
            print(f"Error listing or filtering files in folder {state['image_folder']}: {e}")
            state['is_processing'] = False
            try: state['frame_queue'].put(None)
            except queue.Full: pass
            return
        # --- End: Get image files list with more debugging ---


        # Check if any files were found after filtering
        if not state['image_files']:
            print(f"Error: No image files found in {state['image_folder']} after filtering by extension. Processing cannot start.")
            state['is_processing'] = False
            # Signal end of stream
            try: state['frame_queue'].put(None)
            except queue.Full: pass
            return

        # If we found files, proceed with processing
        print(f"Starting processing loop for {len(state['image_files'])} files.")

        # --- Start: Processing loop ---
        with torch.no_grad():
            # Use the original list directly for the loop
            for i, image_path in enumerate(state['image_files']):

                if not state['is_processing']: # Check stop flag
                    print("Processing stopped by user (stop flag).")
                    break

                frame_start_time = time.time()

                try:
                    image_pil = Image.open(image_path).convert("RGB")
                    # print(f"Successfully opened image: {image_path}") # Keep this debug print if needed
                except Exception as e:
                    print(f"Skipping file {image_path}: Cannot open or process image ({e})")
                    continue

                # 1. Preprocess image
                input_tensor, orig_dims, new_dims = preprocess_image(image_pil, state['max_size'])
                state['orig_dims'] = orig_dims
                state['new_dims'] = new_dims
                # print(f"Frame {i}: Preprocessed. Dims: Orig {orig_dims}, New {new_dims}") # Keep this debug print if needed


                # 2. Run model (Dehaze + Detect)
                # Use the model instance from the state
                dehazed_tensor = state['model'].dehaze(input_tensor)
                dehazed_np = (dehazed_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                dehazed_np = cv2.cvtColor(dehazed_np, cv2.COLOR_RGB2BGR)
                # print(f"Frame {i}: Dehazed.") # Keep this debug print if needed

                predictions = state['model'].predict(input_tensor)
                # print(f"Frame {i}: Predictions obtained. Type: {type(predictions)}, Shape (if array/tensor): {getattr(predictions, 'shape', 'N/A')}") # Keep this debug print if needed


                # 3. Scale boxes to original dimensions
                # Ensure you are using the corrected scale_boxes_to_original function here!
                scaled_predictions = scale_boxes_to_original(predictions, state['orig_dims'], state['new_dims'])
                # print(f"Frame {i}: Scaled predictions. Count: {len(scaled_predictions)}") # Keep this debug print if needed


                # 4. Draw results on the dehazed image
                img_with_boxes = draw_boxes_and_labels(dehazed_np, scaled_predictions)
                # print(f"Frame {i}: Boxes drawn.") # Keep this debug print if needed

                # 5. Draw FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0
                state['current_fps'] = current_fps # Update shared FPS state
                img_with_fps = draw_fps(img_with_boxes, current_fps)
                # print(f"Frame {i}: FPS drawn ({current_fps:.2f})") # Keep this debug print if needed


                # 6. Encode frame to JPEG and put in queue
                # Check the return value of imencode!
                ret, jpeg_frame = cv2.imencode('.jpg', img_with_fps)
                # print(f"Frame {i}: cv2.imencode returned {ret}") # Keep this CRITICAL check

                if not ret:
                    print(f"Error encoding frame {i} to JPEG. Skipping frame.")
                    continue # Skip putting this frame if encoding failed

                # Put frame in queue. Use block=True to ensure it waits if the streamer is slow
                try:
                    # print(f"Frame {i}: Putting frame into queue (size before put: {state['frame_queue'].qsize()})...") # Keep this
                    state['frame_queue'].put(jpeg_frame.tobytes(), block=True) # <-- Use block=True
                    # print(f"Frame {i}: Successfully put frame into queue (size after put: {state['frame_queue'].qsize()})") # Keep this
                except Exception as put_error:
                     print(f"Error putting frame {i} into queue: {put_error}")
                     continue # Just skip the frame for now

                # 7. FPS Control
                frame_process_time = time.time() - frame_start_time
                target_frame_time = 1.0 / state['output_fps']
                time_to_sleep = 0  # Initialize time_to_sleep to 0

                if frame_process_time < target_frame_time:
                    time_to_sleep = target_frame_time - frame_process_time
                    # print(f"Frame {i}: Sleeping for {time_to_sleep:.4f}s") # <-- Move print inside the if
                    time.sleep(time_to_sleep)
                # else:
                # print(f"Frame {i}: Processing took {frame_process_time:.4f}s, slower than target {target_frame_time:.4f}s")

                print(f"Frame {i}: Successfully opened image: {image_path}")

                print(f"Frame {i}: Preprocessed. Dims: Orig {orig_dims}, New {new_dims}")

                print(f"Frame {i}: Dehazed.")

                print(
                    f"Frame {i}: Predictions obtained. Type: {type(predictions)}, Shape (if array/tensor): {getattr(predictions, 'shape', 'N/A')}")

                print(f"Frame {i}: Scaled predictions. Count: {len(scaled_predictions)}")

                print(f"Frame {i}: Boxes drawn.")

                print(f"Frame {i}: FPS drawn ({current_fps:.2f})")

                print(f"Frame {i}: cv2.imencode returned {ret}")

                print(f"Frame {i}: Putting frame into queue (size before put: {state['frame_queue'].qsize()})...")

                print(f"Frame {i}: Successfully put frame into queue (size after put: {state['frame_queue'].qsize()})")

                print(f"Frame {i}: Sleeping for {time_to_sleep:.4f}s")
        # --- End: Processing loop ---


        print("Processing loop finished.")

    except Exception as e:
        print(f"An error occurred during processing thread: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Processing thread finally block entered.")
        state['is_processing'] = False
        # Ensure queue is cleared and signal end
        # print("Clearing queue before sending stop signal...")
        while not state['frame_queue'].empty():
            try: state['frame_queue'].get_nowait()
            except queue.Empty: pass

        # Now, send the stop signal
        # print("Putting None (stop signal) into queue...")
        try:
            state['frame_queue'].put(None, block=True)
            # print("Stop signal successfully put.")
        except Exception as put_none_error:
             print(f"Error putting stop signal into queue: {put_none_error}")

        print("Processing thread finished.")




# Also add a print in the generator to see if it's getting the signal
def generate_mjpeg_stream(frame_queue):
    """Generates an MJPEG stream from frames in the queue."""
    print("MJPEG stream generator started.")
    try: # <-- Add try block
        while True:
            print("Streamer waiting for frame...")
            frame_bytes = frame_queue.get()

            print(f"Streamer got item from queue. Is it None? {frame_bytes is None}")

            if frame_bytes is None: # End signal from processing thread
                print("Received end signal (None) in MJPEG stream generator.")
                break

            print(f"Streamer yielding frame (size: {len(frame_bytes)} bytes).")

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                   b'\r\n' + frame_bytes + b'\r\n')

    except Exception as e: # <-- Catch exceptions
        print(f"An error occurred in MJPEG stream generator: {e}")
        import traceback
        traceback.print_exc()

    finally: # <-- Add finally block
        print("MJPEG stream generator finally block entered.")
        # Ensure any resources are cleaned up if needed
        # The queue is handled by the processing thread

    print("MJPEG stream generator finished.")






# ---- Flask Routes ----

@app.route('/')
def index():
    """Main page."""
    # Pass default values to the template
    default_image_folder = 'data/UAV-M/MiDaS_Deep_UAV-benchmark-M_fog_050/M1005' # Example relative path
    default_yaml_path = 'configs/DE_NET.yaml' # Example relative path
    return render_template('index.html',
                           default_image_folder=default_image_folder,
                           default_yaml_path=default_yaml_path)

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """Receives paths, loads model, starts processing thread."""
    if state['is_processing']:
        return jsonify({"status": "error", "message": "Processing already in progress."})

    user_image_folder_input = request.form.get('image_folder')
    user_yaml_path_input = request.form.get('yaml_path')

    if not user_image_folder_input or not user_yaml_path_input:
        return jsonify({"status": "error", "message": "Image folder and YAML path are required."})

    # Resolve paths relative to the base directory
    resolved_image_folder = resolve_path(user_image_folder_input)
    resolved_yaml_path = resolve_path(user_yaml_path_input)

    if not os.path.isdir(resolved_image_folder):
         return jsonify({"status": "error", "message": f"Image folder not found or is not a directory: {resolved_image_folder}"})

    if not os.path.isfile(resolved_yaml_path):
         return jsonify({"status": "error", "message": f"YAML file not found: {resolved_yaml_path}"})

    # Store resolved paths in state
    state['image_folder'] = resolved_image_folder
    state['yaml_path'] = resolved_yaml_path
    state['model'] = None # Reset model to force reload with new config/path

    # Clear the queue before starting
    while not state['frame_queue'].empty():
        try:
            state['frame_queue'].get_nowait()
        except queue.Empty:
            pass

    # Start the processing thread
    state['processing_thread'] = threading.Thread(target=process_video_thread)
    state['processing_thread'].daemon = True # Allow main thread to exit even if worker is running
    state['processing_thread'].start()

    print("Processing started with resolved paths.")
    return jsonify({"status": "success", "message": "Processing started."})

@app.route('/video_feed')
def video_feed():
    """Provides the MJPEG stream of processed frames."""
    # It's okay if processing hasn't started yet, the generator will wait for frames
    # or the end signal.
    return Response(generate_mjpeg_stream(state['frame_queue']),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    """Stops the processing thread."""
    if state['is_processing']:
        state['is_processing'] = False # Signal the thread to stop
        # The thread's loop checks this flag and exits gracefully.
        # The 'finally' block in the thread will put None into the queue to signal the stream to stop.
        print("Stop signal sent to processing thread.")
        return jsonify({"status": "success", "message": "Processing stop signal sent."})
    else:
        return jsonify({"status": "info", "message": "Processing is not running."})

@app.route('/status')
def get_status():
    """Provides current processing status and FPS."""
    return jsonify({
        "is_processing": state['is_processing'],
        "current_fps": state['current_fps']
    })

# ---- Run the App ----
if __name__ == '__main__':
    # Ensure default torch home exists if needed
    os.makedirs('./.torch', exist_ok=True)
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

