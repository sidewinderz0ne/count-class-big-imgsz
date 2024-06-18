import os
import csv
import torch  # Import torch module to catch torch.cuda.OutOfMemoryError
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
    weights_paths.append(os.path.join(base_dir, 'best.pt'))
    weights_paths.append(os.path.join(base_dir, 'last.pt'))

# Define the image paths
image_paths = [
    '/home/sdz/MRE FOTO UDARA JPG/mre 1 blok.jpg',
    '/home/sdz/MRE FOTO UDARA JPG/mre stngh blok.jpg'
    # Add more image paths here
]

# Determine maximum image size based on GPU memory and model requirements
max_imgsz = 10000  # Example maximum size (adjust based on your GPU's capabilities)
min_imgsz = 352  # Minimum size based on YOLOv5 requirements

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

# Loop through each weight, image, image size, IoU threshold, and confidence level, perform inference, and save results to CSV
for image_path in image_paths:
    csv_path = os.path.splitext(image_path)[0] + ".csv"
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['model', 'imgsz', 'iou', 'conf'] + [f'class_{cls}' for cls in range(2)] + ['total_detections']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for weight_path in weights_paths:
            for iou_thresh in iou_thresholds:
                for conf_level in confidence_levels:
                    image_sizes = range(min_imgsz, max_imgsz + 1, 500)  # Increment by 500, up to max_imgsz
                    for image_size in image_sizes:
                        if image_size % 32 != 0:
                            image_size = (image_size // 32) * 32 + 32  # Ensure multiple of 32
                        
                        detections_per_class, total_detections = perform_inference(weight_path, image_path, image_size, iou_thresh, conf_level, max_det, save_results, show_labels)
                        
                        if detections_per_class is None:
                            continue  # Skip writing to CSV if there was an error
                        
                        row = {
                            'model': os.path.basename(weight_path),
                            'imgsz': image_size,
                            'iou': iou_thresh,
                            'conf': conf_level,
                            'class_0': detections_per_class.get(0, 0),
                            'class_1': detections_per_class.get(1, 0),
                            'total_detections': total_detections
                        }
                        writer.writerow(row)
