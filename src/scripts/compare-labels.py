import os

from PIL import Image

from src.yolo.entities.validation import evaluate_predictions, meanAveragePrecision
from src.yolo.entities.yolo_ensembling_framework import YOLOEnsemble


def get_original_and_new_labels(img_path, original_label_folder, new_label_folder, modified_files_log, classes):
    original_annotations = []
    new_annotations = []

    # Read the modified files log from the provided path
    with open(modified_files_log, 'r') as file:
        modified_files = {os.path.splitext(line.strip())[0] for line in file}

    print("Actual Errors:")
    print(len(modified_files))
    # Create sets of filenames (without extensions) in the original and new label folders
    original_label_filenames = {
        os.path.splitext(filename)[0]
        for filename in os.listdir(original_label_folder)
        if filename.endswith('.txt')
    }

    new_label_filenames = {
        os.path.splitext(filename)[0]
        for filename in os.listdir(new_label_folder)
        if filename.endswith('.txt')
    }

    print("Detected Errors")
    print(len(new_label_filenames))

    # Find common filenames in all three sets
    common_filenames = original_label_filenames & new_label_filenames & modified_files

    for filename in common_filenames:
        original_file_path = os.path.join(original_label_folder, filename + '.txt')
        new_file_path = os.path.join(new_label_folder, filename + '.txt')

        # Get corresponding image dimensions
        image_filename = filename + '.jpg'  # Assuming images have .jpg extension
        image_path = os.path.join(img_path, image_filename)
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Convert YOLO labels to predictions with correct bounding box coordinates for original labels
        original_predictions = YOLOEnsemble.convert_to_predictions(original_file_path, image_width, image_height, classes)
        original_annotations.append(original_predictions)

        # Convert YOLO labels to predictions with correct bounding box coordinates for new labels
        new_predictions = YOLOEnsemble.convert_to_predictions(new_file_path, image_width, image_height, classes)
        new_annotations.append(new_predictions)

    return original_annotations, new_annotations

classes = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "diningtable",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "pottedplant",
    16: "sheep",
    17: "sofa",
    18: "split1",
    19: "tvmonitor"
}

original_labels_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL\labels'
img_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL\images'

new_labels_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\\test\\threshold\\both-50\\new label'
modified_files_log = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\PASCAL\PASCAL Noise\PASCAL Both Noise\\Noise-50\modified_files_log.txt'

org_preds, new_preds = get_original_and_new_labels(img_folder, original_labels_folder, new_labels_folder, modified_files_log, classes)

print("True Errors:")
print(len(org_preds))
print(len(new_preds))

precision, recall, f1_score = evaluate_predictions(new_preds, org_preds)
mAP, classAPs = meanAveragePrecision(new_preds, org_preds, classes)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("mAP Score:", mAP)



