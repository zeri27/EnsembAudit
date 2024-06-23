class Prediction:
    def __init__(self, cls, conf, bbox, actual=False):
        """
        Initializes a Prediction object.

        :param cls: Class label of the prediction.
        :param conf: Confidence score associated with the prediction.
        :param bbox: Bounding box associated with the prediction.
        """
        self.cls = cls
        self.conf = conf
        self.bbox = bbox
        self.actual = actual

    def show(self):
        """
        Returns a string representation of the prediction.

        :return: String containing class and confidence information.
        """
        return "Class: " + self.cls + ", Confidence: " + str(self.conf)


class Bbox:
    def __init__(self, x_min, y_min, x_max, y_max):
        """
        Initializes a Bbox object.

        :param x_min: Minimum x-coordinate of the bounding box.
        :param y_min: Minimum y-coordinate of the bounding box.
        :param x_max: Maximum x-coordinate of the bounding box.
        :param y_max: Maximum y-coordinate of the bounding box.
        """
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def show(self):
        return "x_min: " + str(self.x_min) + ", y_min: " + str(self.y_min) + ", x_max: " + str(self.x_max) + ", y_max: " + str(self.y_max)


def calculate_iou(bbox1, bbox2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    :param bbox1: First bounding box.
    :param bbox2: Second bounding box.
    :return: IoU score between the two bounding boxes.
    """
    x_left = max(bbox1.x_min, bbox2.x_min)
    y_top = max(bbox1.y_min, bbox2.y_min)
    x_right = min(bbox1.x_max, bbox2.x_max)
    y_bottom = min(bbox1.y_max, bbox2.y_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1.x_max - bbox1.x_min) * (bbox1.y_max - bbox1.y_min)
    bbox2_area = (bbox2.x_max - bbox2.x_min) * (bbox2.y_max - bbox2.y_min)
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou

