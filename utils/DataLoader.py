import os
import torch
import yaml
import random
import importlib.util
from glob import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw # Import ImageDraw

def custom_collate_fn(batch):
    """
    Custom collate function to handle batches of images and variable-length label lists.

    Args:
        batch (list): A list of tuples, where each tuple contains (image, labels).
                      image is a Tensor [C, H, W].
                      labels is a list of lists, e.g., [[label1_data], [label2_data], ...].

    Returns:
        tuple: A tuple containing:
               - images (torch.Tensor): A batch of images stacked along the batch dimension [B, C, H, W].
               - labels (list[torch.Tensor]): A list of tensors, where each tensor contains the labels
                                              for the corresponding image [N_i, 9]. N_i is the number of
                                              objects in the i-th image.
    """
    images, labels = zip(*batch)  # list of images, list of label lists
    images = torch.stack(images, dim=0)  # Stack images into a batch tensor [B, C, H, W]
    # Convert each list of label data into a tensor
    labels = [torch.tensor(label, dtype=torch.float32) for label in labels] # list of [N_i, 9] tensors
    return images, labels


class UAVOriginalDataset(Dataset):
    """
    Dataset class for loading UAV image frames and their corresponding annotations.
    Optionally applies masking based on ignore files.
    """
    def __init__(self, image_files, label_files, ignore_files, is_mask, transform=None):
        """
        Initializes the dataset.

        Args:
            image_files (list[str]): List of paths to image files.
            label_files (list[str]): List of paths to annotation label files.
            ignore_files (list[str]): List of paths to ignore region files.
            is_mask (bool): If True, apply masking using ignore files.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.image_files = image_files
        self.label_files = label_files
        self.ignore_files = ignore_files # Store ignore file paths
        self.is_mask = is_mask           # Store masking flag
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Gets the data sample for a given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                   - img (torch.Tensor or PIL.Image): The image (transformed if transform is provided).
                   - annotations (list[list[float]]): A list of annotations for the image.
                                                     Each annotation is a list of floats.
        """
        # Load image
        img = Image.open(self.image_files[idx]).convert("RGB")

        # Apply masking if enabled
        if self.is_mask:
            ignore_path = self.ignore_files[idx]
            if os.path.exists(ignore_path):
                try:
                    draw = ImageDraw.Draw(img) # Create drawing context
                    with open(ignore_path, 'r') as f:
                        for line in f:
                            try:
                                fields = list(map(float, line.strip().split(',')))
                                # Assuming format: bbox_left, bbox_top, bbox_width, bbox_height, ...
                                if len(fields) >= 4:
                                    x1, y1 = fields[0], fields[1]
                                    # Calculate bottom-right coordinates
                                    x2 = x1 + fields[2]
                                    y2 = y1 + fields[3]
                                    # Draw a white rectangle to mask the ignore region
                                    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
                            except ValueError:
                                print(f"Warning: Skipping invalid line in ignore file {ignore_path}: {line.strip()}")
                                continue
                    del draw # Release drawing context
                except Exception as e:
                    print(f"Error processing ignore file {ignore_path}: {e}")


        # Load labels
        label_path = self.label_files[idx]
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        fields = list(map(float, line.strip().split(',')))  # Convert string fields to float
                        annotations.append(fields)
                    except ValueError:
                        print(f"Warning: Skipping invalid line in label file {label_path}: {line.strip()}")
                        continue

        # Apply transformations AFTER masking (if any)
        if self.transform:
            img = self.transform(img)

        return img, annotations  # annotations: List[List[float]]


class UAVDataLoaderBuilder:
    """
    Builds DataLoaders for the UAV dataset, handling data splitting,
    optional haze/dehaze processing, and label extraction.
    """
    def __init__(self, config):
        """
        Initializes the builder with configuration settings.

        Args:
            config (dict): Configuration dictionary containing dataset paths, processing options, etc.
        """
        self.seed = config['seed']
        self.root = config['dataset']['path']
        self.haze_method = config.get('haze_method', "None") # Use .get for safer access
        self.dehaze_method = config.get('dehaze_method', "None")
        self.is_mask = config.get('is_mask', False) # Get masking flag, default to False
        self.image_root = os.path.join(self.root, config['dataset']['data_path'])
        self.label_root = os.path.join(self.root, config['dataset']['label_path'])
        self.is_clean = config.get('is_clean', False) # Get clean dataset flag
        random.seed(self.seed)
        torch.manual_seed(self.seed) # Also seed torch for reproducibility if needed

        # Apply haze/dehaze processing if specified
        self.image_root = self.apply_processing(self.image_root)

    def apply_processing(self, dataset_path):
        """Applies haze or dehaze processing by calling external functions."""
        path = dataset_path
        # Example for haze, extend similarly for dehaze if needed
        if self.haze_method != "None":
            path = self.call_processing_function(path, self.haze_method, 'haze')
        # Add similar block for self.dehaze_method if required
        # if self.dehaze_method != "None":
        #     path = self.call_processing_function(path, self.dehaze_method, 'dehaze')
        return path

    def call_processing_function(self, input_path, method_name, module_prefix):
        """Dynamically imports and calls a processing function."""
        try:
            # Dynamically import the module (e.g., 'haze.method_name' or 'dehaze.method_name')
            module = importlib.import_module(f'{module_prefix}.{method_name}')
            # Get the function with the same name as the method
            func = getattr(module, method_name)
            print(f"Applying {method_name} processing from {module_prefix} module to path: {input_path}")
            # Call the function and return the potentially modified path or processed data root
            return func(input_path)
        except ModuleNotFoundError:
            print(f"Error: Module '{module_prefix}.{method_name}' not found.")
            return input_path # Return original path if module not found
        except AttributeError:
            print(f"Error: Function '{method_name}' not found in module '{module_prefix}.{method_name}'.")
            return input_path # Return original path if function not found
        except Exception as e:
            print(f"Error during {method_name} processing: {e}")
            return input_path # Return original path on other errors

    def parse_labels_to_frames(self, label_file):
        """Parses a ground truth file into a dictionary mapping frame numbers to annotations."""
        frame_map = {}
        if not os.path.exists(label_file):
            print(f"Warning: Label file not found: {label_file}")
            return frame_map
        with open(label_file, 'r') as f:
            for line in f:
                try:
                    items = line.strip().split(',')
                    frame = int(items[0]) # Frame number is the first item
                    # Store the whole line (or parse specific fields if needed later)
                    frame_map.setdefault(frame, []).append(line.strip())
                except (ValueError, IndexError):
                    print(f"Warning: Skipping invalid line in label file {label_file}: {line.strip()}")
                    continue
        return frame_map

    def extract_labels(self, video_folder, label_path, ignore_label_path, save_folder):
        """
        Extracts frame-specific labels from a video's ground truth file and saves them.
        Also prepares the list of corresponding ignore files.

        Args:
            video_folder (str): Path to the folder containing video frames (images).
            label_path (str): Path to the ground truth file for the whole video.
            ignore_label_path (str): Path to the ignore file for the whole video.
            save_folder (str): Path to the directory where frame-specific label files will be saved.

        Returns:
            tuple: Contains:
                   - image_files (list[str]): Sorted list of image file paths for the video.
                   - label_files (list[str]): List of paths to the generated frame-specific label files.
                   - ignore_files (list[str]): List of paths to the corresponding ignore file (repeated for each frame).
        """
        os.makedirs(save_folder, exist_ok=True)
        frame_map = self.parse_labels_to_frames(label_path)
        # Find all jpg images in the video folder and sort them
        image_files = sorted(glob(os.path.join(video_folder, '*.jpg')))
        label_files = []
        ignore_files = [] # List to store ignore file path for each frame

        for i, img_path in enumerate(image_files, start=1): # Frame numbers usually start from 1
            frame_base_name = os.path.basename(img_path)
            label_file_name = frame_base_name.replace('.jpg', '.txt')
            label_file_path = os.path.join(save_folder, label_file_name)

            # Write annotations for the current frame (i) to its specific file
            with open(label_file_path, 'w') as f:
                # Get annotations for frame 'i', default to empty list if none
                for line in frame_map.get(i, []):
                    f.write(line + '\n')

            label_files.append(label_file_path)
            # Add the single video-level ignore file path for each frame
            ignore_files.append(ignore_label_path)

        return image_files, label_files, ignore_files

    def build(self, train_ratio=0.7, val_ratio=0.2, transform=None):
        """
        Builds train, validation, and test datasets (and optionally a clean dataset).

        Args:
            train_ratio (float): Proportion of data to use for training.
            val_ratio (float): Proportion of data to use for validation.
            transform (callable, optional): Transformations to apply to the images. Defaults to None.

        Returns:
            tuple: Depending on self.is_clean, returns either:
                   (train_dataset, val_dataset, test_dataset, clean_dataset) or
                   (train_dataset, val_dataset, test_dataset, None)
        """
        # List video folders (assuming each folder is a video sequence)
        video_folders = sorted([d for d in os.listdir(self.image_root) if os.path.isdir(os.path.join(self.image_root, d))])
        if not video_folders:
             raise FileNotFoundError(f"No video folders found in {self.image_root}")

        random.shuffle(video_folders) # Shuffle videos for splitting
        n = len(video_folders)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split video folders into sets
        train_set_vids = video_folders[:train_end]
        val_set_vids = video_folders[train_end:val_end]
        test_set_vids = video_folders[val_end:]

        datasets = {}
        # Process each split (train, val, test)
        for split_name, video_list in [('train', train_set_vids), ('val', val_set_vids), ('test', test_set_vids)]:
            all_imgs, all_labels, all_ignores = [], [], []
            print(f"Processing {split_name} set...")
            for vid in video_list:
                vid_folder = os.path.join(self.image_root, vid)
                # Construct paths for the whole video's ground truth and ignore files
                label_file = os.path.join(self.label_root, f"{vid}_gt_whole.txt")
                ignore_file = os.path.join(self.label_root, f"{vid}_gt_ignore.txt") # Path to ignore file
                # Define where to save frame-specific labels
                label_save_folder = os.path.join(self.root, 'frame_labels', split_name, vid)

                # Extract frame images, labels, and corresponding ignore file paths
                imgs, labels, ignores = self.extract_labels(vid_folder, label_file, ignore_file, label_save_folder)

                all_imgs.extend(imgs)
                all_labels.extend(labels)
                all_ignores.extend(ignores) # Add ignore file paths

            if not all_imgs:
                 print(f"Warning: No images found for the {split_name} split.")
                 datasets[split_name] = None # Or handle as an error
            else:
                 # Create the dataset for the current split, passing ignore files and mask flag
                 datasets[split_name] = UAVOriginalDataset(all_imgs, all_labels, all_ignores, self.is_mask, transform)

        # Process clean dataset if requested
        clean_dataset = None
        if self.is_clean:
            all_imgs_clean, all_labels_clean, all_ignores_clean = [], [], []
            print("Processing clean set...")
            # Assuming 'clean' uses the same videos as 'train' but maybe different processing/transforms later
            clean_set_vids = train_set_vids
            for vid in clean_set_vids:
                vid_folder = os.path.join(self.image_root, vid) # Use the potentially processed image root
                label_file = os.path.join(self.label_root, f"{vid}_gt_whole.txt")
                ignore_file = os.path.join(self.label_root, f"{vid}_gt_ignore.txt")
                # Save frame labels in a separate 'clean' directory
                label_save_folder = os.path.join(self.root, 'frame_labels', 'clean', vid)
                imgs, labels, ignores = self.extract_labels(vid_folder, label_file, ignore_file, label_save_folder)
                all_imgs_clean.extend(imgs)
                all_labels_clean.extend(labels)
                all_ignores_clean.extend(ignores)

            if not all_imgs_clean:
                 print(f"Warning: No images found for the clean split.")
            else:
                # Create the clean dataset
                clean_dataset = UAVOriginalDataset(all_imgs_clean, all_labels_clean, all_ignores_clean, self.is_mask, transform)

        return datasets.get('train'), datasets.get('val'), datasets.get('test'), clean_dataset

