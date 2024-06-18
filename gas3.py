import os
import csv
import torch
from datetime import datetime
from ultralytics import YOLO

# Define the base weight directories
base_weight_dirs = [
    '/home/sdz/MRE FOTO UDARA JPG/training_111_sampel/2024-06-14-07-24_yolov8s_1280_111_pokok_kuning/train/weights/',
    '/home/sdz/MRE FOTO UDARA JPG/training_111_sampel/2024-06-14-14-37_yolov8m_1280_111_pokok_kuning/train/weights/',
    '/home/sdz/MRE FOTO UDARA JPG/training_111_sampel/2024-06-14-21-49_yolov8l_1280_111_pokok_kuning/train/weights/',
    '/home/sdz/MRE FOTO UDARA JPG/training_111_sampel/2024-06-15-05-00_yolov8x_1280_111_pokok_kuning/train/weights/',
    '/home/sdz/MRE FOTO UDARA JPG/training_111_sampel/2024-06-15-16-33_yolov5nu_1280_111_pokok_kuning/train/weights/',
    '/home/sdz/MRE FOTO UDARA JPG/training_111_sampel/2024-06-15-18-17_yolov5mu_1280_111_pokok_kuning/train/weights/',
    '/home/sdz/MRE FOTO UDARA JPG/training_111_sampel/2024-06-16-01-41_yolov5lu_1280_111_pokok_kuning/train/weights/',
    '/home/sdz/MRE FOTO UDARA JPG/training_111_sampel/2024-06-16-02-12_yolov5xu_1280_111_pokok_kuning/train/weights/'
    # Add more base weight directories here
]

# Automatically add 'best.pt' and 'last.pt' for each base directory
weights_paths = []
for base_dir in base_weight_dirs:
    weight_files = [f for f in os.listdir(base_dir) if f.endswith('.pt')]
    for weight_file in weight_files:
        weights_paths.append(os.path.join(base_dir, weight_file))

# Define the image directory
image_dir = '/home/sdz/MRE FOTO UDARA JPG/'

# List all image files in the image directory
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

# Define the minimum and maximum image size
min_imgsz = 352  # Minimum size based on YOLOv5 requirements
max_imgsz = 12000  # Example maximum size (adjust based on your GPU's capabilities)
imgsz_step = 500  # Step size for decreasing image size

# Define the IoU thresholds and confidence levels to test
iou_thresholds = [0.25, 0.5, 0.75]  # Add more IoU thresholds as needed
confidence_levels = [0.01, 0.05, 0.1]  # Add more confidence levels as needed

# Define the configuration parameters
max_det = 12000
save_results = True
show_labels = False

# Function to perform inference using YOLOv8 and return the number of detected objects for each class
def perform_inference(weight_path, image_path, image_size, iou_thresh, conf_level, max_det, save_results, show_labels):
    model = YOLO(model=weight_path)
    try:
        results = model.predict(source=image_path, imgsz=image_size, iou=iou_thresh, conf=conf_level, max_det=max_det, save=save_results, show_labels=show_labels)
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA out of memory error occurred: {e}")
        return None, None  # Return None to indicate failure
    
    detections_per_class = {cls: 0 for cls in range(len(model.names))} if results else None
    total_detections = 0

    if results:
        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    cls = int(box.cls)
                    detections_per_class[cls] += 1
                    total_detections += 1

    return detections_per_class, total_detections

# Loop through each image, each weight, IoU threshold, and confidence level, perform inference, and append results to CSV
for image_path in image_files:
    csv_path = os.path.splitext(image_path)[0] + ".csv"
    with open(csv_path, mode='a', newline='') as csv_file:
        fieldnames = ['model', 'imgsz', 'iou', 'conf'] + [f'class_{cls}' for cls in range(2)] + ['total_detections', 'timestamp']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Check if file is empty, if so, write header
        if os.stat(csv_path).st_size == 0:
            writer.writeheader()
        
        for weight_path in weights_paths:
            model_name = os.path.basename(os.path.dirname(weight_path))
            for iou_thresh in iou_thresholds:
                for conf_level in confidence_levels:
                    image_size = max_imgsz
                    while image_size >= min_imgsz:
                        try:
                            detections_per_class, total_detections = perform_inference(weight_path, image_path, image_size, iou_thresh, conf_level, max_det, save_results, show_labels)
                            if detections_per_class is not None:
                                timestamp = datetime.now().isoformat()
                                row = {
                                    'model': weight_path,
                                    'imgsz': image_size,
                                    'iou': iou_thresh,
                                    'conf': conf_level,
                                    'class_0': detections_per_class.get(0, 0),
                                    'class_1': detections_per_class.get(1, 0),
                                    'total_detections': total_detections,
                                    'timestamp': timestamp
                                }
                                writer.writerow(row)
                                break  # Break out of the while loop if successful
                            else:
                                image_size -= imgsz_step  # Decrease image size by step if CUDA out of memory
                        except Exception as e:
                            print(f"Error occurred during inference: {e}")
                            break  # Break out of the while loop on any error
