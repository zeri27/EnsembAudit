from src.yolo.entities.utility import calculate_iou, Bbox, Prediction


class Average:
    def __init__(self):
        pass

    @staticmethod
    def ensemble_predictions_averaging(predictions_list, iou_threshold=0.5):
        ensemble_results = []
        for image_prediction in predictions_list:
            combined_predictions = []
            for model_prediction in image_prediction:
                for prediction in model_prediction:
                    combined_predictions.append(prediction)

            averaged_predictions = Average.average_predictions(combined_predictions, iou_threshold)
            ensemble_results.append(averaged_predictions)

        return ensemble_results

    @staticmethod
    def average_predictions(predictions, iou_threshold=0.5):
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

        averaged_predictions = []
        for group in grouped_predictions:
            if len(group) == 1:
                averaged_predictions.append(group[0])
                continue

            avg_cls = group[0].cls  # Assuming all predictions in the group have the same class
            avg_conf = sum(pred.conf for pred in group) / len(group)
            avg_bbox = Bbox(
                x_min=sum(pred.bbox.x_min for pred in group) / len(group),
                y_min=sum(pred.bbox.y_min for pred in group) / len(group),
                x_max=sum(pred.bbox.x_max for pred in group) / len(group),
                y_max=sum(pred.bbox.y_max for pred in group) / len(group)
            )

            averaged_predictions.append(Prediction(avg_cls, avg_conf, avg_bbox))

        return averaged_predictions