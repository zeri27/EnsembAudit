from collections import Counter

from src.yolo.entities.utility import calculate_iou


class Fusion:
    def __init__(self):
        pass

    @staticmethod
    def ensemble_predictions_fusion(predictions_list):
        ensemble_results = []
        for image_prediction in predictions_list:
            #disagreement = Fusion.check_disagreement(image_prediction)
            disagreement = True
            # Disagreement needs to always be true when checking for ensemble performance
            #print(disagreement)
            if disagreement:
                # Step 1: Combine predictions from all models for the current image
                combined_predictions = []
                for model_prediction in image_prediction:
                    for prediction in model_prediction:
                        combined_predictions.append(prediction)

                # Step 2: Apply Non-Maximum Suppression
                final_predictions = Fusion.non_max_suppression(combined_predictions)
                ensemble_results.append(final_predictions)
            else:
                ensemble_results.append(None)

        return ensemble_results

    @staticmethod
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

    @staticmethod
    def check_disagreement(image_predictions):
        prediction_groups = []

        # Group similar predictions together
        for model_preds in image_predictions:
            for pred in model_preds:
                matched = False
                for group in prediction_groups:
                    if pred.cls == group[0].cls and calculate_iou(pred.bbox, group[0].bbox) >= 0.5:
                        group.append(pred)
                        matched = True
                        break
                if not matched:
                    prediction_groups.append([pred])

        # Count the number of models supporting each group
        group_counts = [len(group) for group in prediction_groups]

        # Check if the most supported group has a majority
        num_models = len(image_predictions)
        most_supported_group_count = max(group_counts, default=0)

        # THRESHOLD VOTING
        # We use 5 models, so 4+ models need to accept else there is a disagreement
        if most_supported_group_count >= 4:
            return False  # No disagreement
        else:
            return True  # Disagreement

