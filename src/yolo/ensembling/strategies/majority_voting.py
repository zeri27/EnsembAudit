from collections import Counter

from src.yolo.entities.utility import calculate_iou, Bbox, Prediction


class MajorityVoting:
    def __init__(self):
        pass

    @staticmethod
    def ensemble_predictions_majority_voting(predictions_list, iou_threshold=0.5):
        ensemble_results = []
        for image_prediction in predictions_list:
            combined_predictions = []
            for model_prediction in image_prediction:
                for prediction in model_prediction:
                    combined_predictions.append(prediction)

            majority_voted_predictions = MajorityVoting.majority_voting(combined_predictions, iou_threshold)
            ensemble_results.append(majority_voted_predictions)

        return ensemble_results

    @staticmethod
    def majority_voting(predictions, iou_threshold=0.5):
        if len(predictions) == 0:
            return []

        predictions.sort(key=lambda x: x.conf, reverse=True)
        grouped_predictions = []

        while predictions:
            base_pred = predictions.pop(0)
            group = [base_pred]

            predictions_copy = predictions[:]
            for pred in predictions_copy:
                if calculate_iou(base_pred.bbox, pred.bbox) >= iou_threshold:
                    group.append(pred)
                    predictions.remove(pred)

            grouped_predictions.append(group)

        majority_voted_predictions = []
        for group in grouped_predictions:
            if len(group) == 1:
                majority_voted_predictions.append(group[0])
                continue

            class_counter = Counter(pred.cls for pred in group)
            majority_class = class_counter.most_common(1)[0][0]

            avg_conf = sum(pred.conf for pred in group if pred.cls == majority_class) / class_counter[majority_class]
            avg_bbox = Bbox(
                x_min=sum(pred.bbox.x_min for pred in group if pred.cls == majority_class) / class_counter[
                    majority_class],
                y_min=sum(pred.bbox.y_min for pred in group if pred.cls == majority_class) / class_counter[
                    majority_class],
                x_max=sum(pred.bbox.x_max for pred in group if pred.cls == majority_class) / class_counter[
                    majority_class],
                y_max=sum(pred.bbox.y_max for pred in group if pred.cls == majority_class) / class_counter[
                    majority_class]
            )

            majority_voted_predictions.append(Prediction(majority_class, avg_conf, avg_bbox))

        return majority_voted_predictions

