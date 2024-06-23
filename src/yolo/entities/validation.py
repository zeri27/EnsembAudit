import numpy as np

from src.yolo.entities.utility import calculate_iou


def evaluate_predictions(ensemble, ground_truth, iou_threshold=0.5):
    total_predictions = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for image_predictions, image_gt in zip(ensemble, ground_truth):
        if image_predictions is None:
            continue
        total_predictions += len(image_predictions)

        # Create sets to keep track of matched predictions and ground truth
        matched_predictions = set()
        matched_ground_truth = set()

        # Match predictions with ground truth using IoU
        for prediction in image_predictions:
            max_iou = 0
            matched_gt_index = None

            for i, gt_bbox in enumerate(image_gt):
                iou = calculate_iou(prediction.bbox, gt_bbox.bbox)
                if iou > max_iou:
                    max_iou = iou
                    matched_gt_index = i

            if max_iou >= iou_threshold:
                true_positives += 1
                matched_predictions.add(prediction)
                matched_ground_truth.add(matched_gt_index)
            else:
                false_positives += 1

        # Count unmatched ground truth as false negatives
        false_negatives += len(image_gt) - len(matched_ground_truth)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score

def meanAveragePrecision(ensemble, ground_truth, classes, iou_threshold=0.5):
    class_aps = {}

    for class_idx, class_name in classes.items():
        all_true_positives = []
        all_false_positives = []
        all_scores = []
        num_ground_truths = 0

        for image_predictions, image_gt in zip(ensemble, ground_truth):
            if image_predictions is None:
                continue

            class_predictions = [pred for pred in image_predictions if pred.cls == class_name]
            class_gt = [gt for gt in image_gt if gt.cls == class_name]

            num_ground_truths += len(class_gt)

            detected = [False] * len(class_gt)

            for pred in class_predictions:
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(class_gt):
                    iou = calculate_iou(pred.bbox, gt.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and not detected[best_gt_idx]:
                    detected[best_gt_idx] = True
                    all_true_positives.append(1)
                    all_false_positives.append(0)
                else:
                    all_true_positives.append(0)
                    all_false_positives.append(1)

                all_scores.append(pred.conf)

        if num_ground_truths == 0:
            class_aps[class_name] = 0.0
            continue

        all_true_positives = np.array(all_true_positives)
        all_false_positives = np.array(all_false_positives)
        all_scores = np.array(all_scores)

        sorted_indices = np.argsort(-all_scores)
        all_true_positives = all_true_positives[sorted_indices]
        all_false_positives = all_false_positives[sorted_indices]

        cumulative_true_positives = np.cumsum(all_true_positives)
        cumulative_false_positives = np.cumsum(all_false_positives)

        precisions = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives)
        recalls = cumulative_true_positives / num_ground_truths

        precisions = np.concatenate(([0], precisions, [0]))
        recalls = np.concatenate(([0], recalls, [1]))

        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

        indices = np.where(recalls[1:] != recalls[:-1])[0]

        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        class_aps[class_name] = ap

    mAP = sum(class_aps.values()) / len(class_aps)

    return mAP, class_aps