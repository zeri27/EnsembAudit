import os
import shutil

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from PIL import Image, ImageDraw, ImageFont

from src.yolo.ensembling.strategies.average import Average
from src.yolo.ensembling.strategies.fusion import Fusion
from src.yolo.ensembling.strategies.majority_voting import MajorityVoting
from src.yolo.ensembling.strategies.strategy import EnsembleStrategy
from src.yolo.entities.utility import Bbox, Prediction
from src.yolo.entities.validation import evaluate_predictions, meanAveragePrecision


class YOLOEnsemble:
    def __init__(self, model_paths, img_path):
        self.models = [YOLO(path) for path in model_paths]
        self.img_path = img_path

        self.classes = []
        self.all_model_predictions = [[] for _ in range(len(model_paths))]
        self.normalized_model = [[] for _ in range(len(model_paths))]

    def predict_images(self):
        image_files = [os.path.join(self.img_path, filename) for filename in os.listdir(self.img_path) if
                       filename.endswith(('.jpg', '.jpeg', '.png'))]

        # Initialize tqdm with the total number of images
        pbar = tqdm(total=len(image_files), desc="Predicting images")

        # Run inference on each image for each model
        all_predictions = []
        for img_path in image_files:
            predictions = []
            for i, model in enumerate(self.models):
                output = model(img_path)
                individual = [output]
                self.all_model_predictions[i].append(individual)
                predictions.append(output)

            all_predictions.append(predictions)

            # Update the progress bar
            pbar.update(1)

        # Close the progress bar
        pbar.close()

        return all_predictions

    def normalize_predictions(self, all_predictions):
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
                    if len(self.classes) == 0:
                        self.classes = result.names
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

    def drawOriginalLabels(self, labels_folder, ensembled, actual_directory, label_dir):
        annotations = YOLOEnsemble.get_labels(self.img_path, labels_folder, self.classes)
        YOLOEnsemble.draw_org_annotations(self.img_path, ensembled, annotations, actual_directory)
        return annotations
        #YOLOEnsemble.save_org_labels(self, self.img_path, ensembled, annotations, label_dir)

    @staticmethod
    def draw_org_annotations(image_folder, ensemble, original_labels, output_folder):
        """
        Draws annotations on images based on predictions from the ensemble and saves them in the output folder.

        :param image_folder: Path to the folder containing images.
        :param ensemble: List of lists where each sublist contains Prediction objects for an image.
        :param original_labels: List of lists where each sublist contains original Prediction objects for an image.
        :param output_folder: Path to the folder where annotated images will be saved.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                       filename.endswith(('.jpg', '.jpeg', '.png'))]

        for image_path in image_files:
            image_index = image_files.index(image_path)
            predictions = ensemble[image_index]
            original_predictions = original_labels[image_index]

            if predictions is not None and original_predictions is not None:
                image = Image.open(image_path)
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype("arialbd.ttf", size=14)

                for original_prediction in original_predictions:
                    bbox = original_prediction.bbox
                    draw.rectangle([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
                                   outline='red', width=3)
                    draw.text((bbox.x_min, bbox.y_min),
                              f"{original_prediction.cls} {original_prediction.conf:.2f}",
                              fill='red', font=font)

                output_path = os.path.join(output_folder, f"annotated_{os.path.basename(image_path)}")
                image.save(output_path)

    def validate(self, labels_folder, ensemble_model=None, individual_models=False, actual_directory=None):
        annotations = YOLOEnsemble.get_labels(self.img_path, labels_folder, self.classes)
        if actual_directory is not None:
            YOLOEnsemble.draw_annotations(self.img_path, annotations, actual_directory)
        if individual_models:
            for i, preds in enumerate(self.all_model_predictions):
                normalized = self.normalize_predictions(preds)
                self.normalized_model[i] = YOLOEnsemble.normalize_individual(normalized)
            for i, preds in enumerate(self.normalized_model):
                precision, recall, f1_score = evaluate_predictions(preds, annotations)
                mAP, classAPs = meanAveragePrecision(preds, annotations, self.classes)
                print("MODEL " + str(i + 1))
                print("Precision:", precision)
                print("Recall:", recall)
                print("F1-score:", f1_score)
                print("mAP Score:", mAP)
        if ensemble_model is not None:
            precision, recall, f1_score = evaluate_predictions(ensemble_model, annotations)
            mAP, classAPs = meanAveragePrecision(ensemble_model, annotations, self.classes)
            print("ENSEMBLE MODEL")
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1-score:", f1_score)
            print("mAP Score:", mAP)

    @staticmethod
    def ensemble_predictions(strategy, predictions):
        if strategy == EnsembleStrategy.FUSION:
            ensembled = Fusion.ensemble_predictions_fusion(predictions)
        elif strategy == EnsembleStrategy.AVERAGE:
            ensembled = Average.ensemble_predictions_averaging(predictions)
        elif strategy == EnsembleStrategy.MAJORITY_VOTING:
            ensembled = MajorityVoting.ensemble_predictions_majority_voting(predictions)
        else:
            raise ValueError(f"Unknown Strategy: {strategy}")

        return ensembled

    @staticmethod
    def convert_to_predictions(file_path, image_width, image_height, classes):
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

    @staticmethod
    def get_labels(img_path, folder_path, classes):
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
                predictions = YOLOEnsemble.convert_to_predictions(file_path, image_width, image_height, classes)
                annotations.append(predictions)

        return annotations

    @staticmethod
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

    @staticmethod
    def draw_annotations(image_folder, ensemble, output_folder):
        """
        Draws annotations on images based on predictions from the ensemble and saves them in the output folder.

        :param image_folder: Path to the folder containing images.
        :param ensemble: List of lists where each sublist contains Prediction objects for an image.
        :param output_folder: Path to the folder where annotated images will be saved.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                       filename.endswith(('.jpg', '.jpeg', '.png'))]

        for image_path in image_files:
            predictions = ensemble[image_files.index(image_path)]

            if predictions is not None:
                image = Image.open(image_path)
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype("arialbd.ttf", size=14)

                for prediction in predictions:
                    bbox = prediction.bbox
                    draw.rectangle([bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max],
                                   outline='red', width=3)
                    draw.text((bbox.x_min, bbox.y_min), f"{prediction.cls} {prediction.conf:.2f}",
                              fill='red', font=font)

                output_path = os.path.join(output_folder, f"annotated_{os.path.basename(image_path)}")
                image.save(output_path)

    @staticmethod
    def clear_output_folder(output_folder):
        """
        Clears the contents of the output folder.

        :param output_folder: Path to the folder to be cleared.
        """
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    def save_yolo_labels(self, image_folder, ensemble, original_labels, output_folder, old_folder):
        """
        Converts predictions to YOLO format labels and saves them as text files.

        :param image_folder: Path to the folder containing images.
        :param ensemble: List of lists where each sublist contains Prediction objects for an image.
        :param output_folder: Path to the folder where YOLO label files will be saved.
        """
        class_to_index = {v: k for k, v in self.classes.items()}
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                       filename.endswith(('.jpg', '.jpeg', '.png'))]

        for image_path in image_files:
            image = Image.open(image_path)
            width, height = image.size

            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            label_filename = os.path.join(output_folder, f"{base_filename}.txt")

            predictions = ensemble[image_files.index(image_path)]
            original_predictions = original_labels[image_files.index(image_path)]
            if predictions is not None:
                label_old_filename = os.path.join(old_folder, f"{base_filename}.txt")
                with open(label_filename, 'w') as label_file:
                    for prediction in predictions:
                        bbox = prediction.bbox

                        # Calculate YOLO format values
                        x_center = (bbox.x_min + bbox.x_max) / 2 / width
                        y_center = (bbox.y_min + bbox.y_max) / 2 / height
                        bbox_width = (bbox.x_max - bbox.x_min) / width
                        bbox_height = (bbox.y_max - bbox.y_min) / height

                        # Write the label in YOLO format
                        label_file.write(
                            f"{class_to_index.get(prediction.cls)} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

                with open(label_old_filename, 'w') as label_file2:
                    for prediction in original_predictions:
                        bbox = prediction.bbox

                        # Calculate YOLO format values
                        x_center = (bbox.x_min + bbox.x_max) / 2 / width
                        y_center = (bbox.y_min + bbox.y_max) / 2 / height
                        bbox_width = (bbox.x_max - bbox.x_min) / width
                        bbox_height = (bbox.y_max - bbox.y_min) / height

                        # Write the label in YOLO format
                        label_file2.write(
                            f"{class_to_index.get(prediction.cls)} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    def save_org_labels(self, image_folder, ensemble, original_labels, output_folder):
        """
        Converts predictions to YOLO format labels and saves them as text files.

        :param image_folder: Path to the folder containing images.
        :param ensemble: List of lists where each sublist contains Prediction objects for an image.
        :param original_labels: List of lists where each sublist contains original Prediction objects for an image.
        :param output_folder: Path to the folder where YOLO label files will be saved.
        """
        class_to_index = {v: k for k, v in self.classes.items()}
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                       filename.endswith(('.jpg', '.jpeg', '.png'))]

        for image_path in image_files:
            image = Image.open(image_path)
            width, height = image.size

            predictions = ensemble[image_files.index(image_path)]
            original_predictions = original_labels[image_files.index(image_path)]

            if predictions is not None and original_predictions is not None:
                print('reaced')
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                label_filename = os.path.join(output_folder, f"{base_filename}.txt")
                with open(label_filename, 'w') as label_file:
                    for original_prediction in original_predictions:
                        original_bbox = original_prediction.bbox
                        original_x_center = (original_bbox.x_min + original_bbox.x_max) / 2 / width
                        original_y_center = (original_bbox.y_min + original_bbox.y_max) / 2 / height
                        original_bbox_width = (original_bbox.x_max - original_bbox.x_min) / width
                        original_bbox_height = (original_bbox.y_max - original_bbox.y_min) / height

                        # Write the original label in YOLO format
                        label_file.write(
                            f"{class_to_index.get(original_prediction.cls)} {original_x_center:.6f} {original_y_center:.6f} {original_bbox_width:.6f} {original_bbox_height:.6f}\n")