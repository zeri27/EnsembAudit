# FIRST CODE ITERATION (NOT FOR USE)

import os
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

from ultralytics import YOLO
import numpy as np

from tqdm import tqdm

from src.yolo.entities.utility import Bbox, Prediction, calculate_iou
from src.yolo.entities.validation import evaluate_predictions

# Load the YOLO models
model1 = YOLO(
    'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\detect\pretrained-model-split1\weights\\best.pt')
model2 = YOLO(
    'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\detect\pretrained-model-split2\weights\\best.pt')
model3 = YOLO(
    'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\detect\pretrained-model-split3\weights\\best.pt')
model4 = YOLO(
    'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\detect\pretrained-model-train\weights\\best.pt')
model5 = YOLO(
    'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\yolo\\runs\detect\pretrained-model-split5\weights\\best.pt')

models = [model1, model2, model3, model4, model5]
# models = [model2, model3]

all_model_predictions = [[], [], [], [], []]
# all_model_predictions = [[], []]

classes = []


def predict_images(models, images_path):
    image_files = [os.path.join(images_path, filename) for filename in os.listdir(images_path) if
                   filename.endswith(('.jpg', '.jpeg', '.png'))]

    # Initialize tqdm with the total number of images
    pbar = tqdm(total=len(image_files), desc="Predicting images")

    # Run inference on each image for each model
    all_predictions = []
    for img_path in image_files:
        predictions = []
        for i, model in enumerate(models):
            output = model(img_path)
            individual = [output]
            all_model_predictions[i].append(individual)
            predictions.append(output)

        all_predictions.append(predictions)

        # Update the progress bar
        pbar.update(1)

    # Close the progress bar
    pbar.close()

    return all_predictions


def ensemble_predictions_average(predictions_list, num_models=5):
    ensemble_results = []

    for image_predictions in zip(*predictions_list):
        # Initialize a Counter to count the votes for each class
        votes = Counter()

        # Collect votes from each model's prediction
        for model_prediction in image_predictions:
            for prediction in model_prediction:
                votes[prediction.cls] += 1  # Assuming prediction.label contains the class label

        # Determine the final prediction based on the majority of votes
        final_prediction = votes.most_common(1)[0][0]

        ensemble_results.append(final_prediction)

    return ensemble_results


def ensemble_predictions_custom(predictions_list):
    ensemble_results = []
    for image_prediction in predictions_list:
        # Step 1: Combine predictions from all models for the current image
        combined_predictions = []
        for model_prediction in image_prediction:
            for prediction in model_prediction:
                combined_predictions.append(prediction)

        # Step 2: Merge duplicate predictions and aggregate confidence scores
        merged_predictions = merge_duplicate_predictions(combined_predictions)

        # Step 3: Apply Non-Maximum Suppression
        # final_predictions = non_max_suppression(merged_predictions)

        # ensemble_results.append(final_predictions)
        ensemble_results.append(merged_predictions)

    return ensemble_results


def merge_duplicate_predictions(predictions):
    merged_predictions = []

    for pred1 in predictions:
        merged = False
        for pred2 in merged_predictions:
            if calculate_iou(pred1.bbox, pred2.bbox) >= 0.5:
                if pred1.conf > pred2.conf:
                    merged_predictions.remove(pred2)
                    merged_predictions.append(pred1)
                merged = True
                break
        if not merged:
            merged_predictions.append(pred1)

    return merged_predictions


def non_max_suppression(predictions, iou_threshold=0.5):
    """
    Apply non-maximum suppression to filter out redundant predictions.
    """
    if len(predictions) == 0:
        return []

    predictions.sort(key=lambda x: x.conf, reverse=True)
    selected_predictions = []

    for pred in predictions:
        if len(selected_predictions) == 0:
            selected_predictions.append(pred)
        else:
            overlaps = [calculate_iou(pred.bbox, prev_pred.bbox) for prev_pred in selected_predictions]
            if max(overlaps) < iou_threshold:
                selected_predictions.append(pred)

    return selected_predictions


def normalize_predictions(all_predictions):
    global classes
    normalized_all_predictions = []  # size = number of images
    for image_predictions in all_predictions:
        normalized_image_predictions = []  # size = number of models
        for model_prediction in image_predictions:
            normalized_model_predictions = []  # size = number of predicted boxes by model
            for result in model_prediction:
                boxes_class = result.boxes.cls
                boxes_conf = result.boxes.conf
                boxes_tensor = result.boxes.xyxy
                class_np = boxes_class.cpu().numpy()
                conf_np = boxes_conf.cpu().numpy()
                boxes_np = boxes_tensor.cpu().numpy()
                if len(classes) == 0:
                    classes = result.names
                for i in np.ndindex(boxes_np.shape[:-1]):
                    cls = result.names.get(int(class_np[i]))
                    conf = float(conf_np[i])
                    xmin, ymin, xmax, ymax = boxes_np[i]
                    bbox = Bbox(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax)
                    prediction = Prediction(cls, conf, bbox)
                    normalized_model_predictions.append(prediction)
            normalized_image_predictions.append(normalized_model_predictions)
        normalized_all_predictions.append(normalized_image_predictions)
    return normalized_all_predictions


def draw_annotations(image_folder, ensemble, output_folder):
    """
    Draws annotations on images based on predictions from the ensemble and saves them in the output folder.

    :param image_folder: Path to the folder containing images.
    :param ensemble: List of lists where each sublist contains Prediction objects for an image.
    :param output_folder: Path to the folder where annotated images will be saved.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of image files
    image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                   filename.endswith(('.jpg', '.jpeg', '.png'))]

    for image_path in image_files:
        # Load the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Define font and font size
        font = ImageFont.truetype("arialbd.ttf", size=14)

        # Draw annotations for each prediction
        for prediction in ensemble[image_files.index(image_path)]:
            bbox = prediction.bbox
            # Draw bounding box (slightly thicker) with class-specific color
            draw.rectangle([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
                           outline='red', width=3)

            # Add class and confidence text (slightly bigger and bold)
            draw.text((bbox.x_min, bbox.y_min), f"{prediction.cls} {prediction.conf:.2f}",
                      fill='red', font=font)

        # Save annotated image
        output_path = os.path.join(output_folder, f"annotated_{os.path.basename(image_path)}")
        image.save(output_path)


# VALIDATION
def convert_to_predictions(file_path, image_width, image_height):
    predictions = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            cls = classes[int(parts[0])]
            conf = 1

            # Convert YOLO bounding box attributes to x_min, y_min, x_max, y_max
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Calculate bounding box coordinates based on actual image dimensions
            x_min = (x_center - (width / 2)) * image_width
            y_min = (y_center - (height / 2)) * image_height
            x_max = (x_center + (width / 2)) * image_width
            y_max = (y_center + (height / 2)) * image_height

            bbox = Bbox(x_min, y_min, x_max, y_max)

            predictions.append(Prediction(cls, conf, bbox))

    return predictions


def get_labels(folder_path):
    annotations = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            # Get corresponding image dimensions
            image_filename = os.path.splitext(filename)[0] + '.jpg'  # Assuming images have .jpg extension
            image_path = os.path.join(img_path, image_filename)
            with Image.open(image_path) as img:
                image_width, image_height = img.size

            # Convert YOLO labels to predictions with correct bounding box coordinates
            predictions = convert_to_predictions(file_path, image_width, image_height)
            annotations.append(predictions)

    return annotations


img_path = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\VOC-Validation\images\\split1'

predictions = predict_images(models, img_path)

normalized_model = [[], [], [], [], []]
# normalized_model = [[], []]


def normalize_individual(predictions_list):
    ensemble_results = []
    for image_prediction in predictions_list:
        # Step 1: Combine predictions from all models for the current image
        combined_predictions = []
        for model_prediction in image_prediction:
            for prediction in model_prediction:
                combined_predictions.append(prediction)

        ensemble_results.append(combined_predictions)

    return ensemble_results


for i, preds in enumerate(all_model_predictions):
    normalized = normalize_predictions(preds)
    normalized_model[i] = normalize_individual(normalized)

normalized = normalize_predictions(predictions)

ensemble = ensemble_predictions_custom(normalized)

output_dir = '/src/datasets/VOC-Validation/annotations'

draw_annotations(img_path, ensemble, output_dir)

label_folder = 'C:\\Users\zerya\OneDrive\Desktop\TU Delft\Research Project\Impl\src\datasets\VOC-Validation\labels\\split1'

annotations = get_labels(label_folder)

actual_dir = '/src/datasets/VOC-Validation/actual_annotate'

draw_annotations(img_path, annotations, actual_dir)

accuracy, precision, recall, f1_score = evaluate_predictions(ensemble, annotations)
print("ENSEMBLE MODEL")
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)

for i, preds in enumerate(normalized_model):
    accuracy, precision, recall, f1_score = evaluate_predictions(preds, annotations)
    print("MODEL " + str(i + 1))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1_score)
