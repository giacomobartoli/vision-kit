# DISCLAIMER: this code has been taken by Google Vision Kit and edit by me.
"""API for Object Detection tasks."""
import math
import sys
import os

from aiy.vision.inference import ModelDescriptor
from aiy.vision.models import utils
from aiy.vision.models.object_detection_anchors import ANCHORS

_COMPUTE_GRAPH_NAME = os.getcwd() + '/pikachu_detector.binaryproto'
_NUM_ANCHORS = len(ANCHORS)
_MACHINE_EPS = sys.float_info.epsilon


_NUM_LABELS = 2

class Object(object):
    """Object detection result."""
    BACKGROUND = 0
    PIKACHU = 1

    _LABELS = {
        BACKGROUND: 'BACKGROUND',
        PIKACHU: 'PIKACHU',
    }

    def __init__(self, bounding_box, kind, score):
        """Initialization.
        Args:
          bounding_box: a tuple of 4 ints, (x, y, width, height) order.
          kind: int, tells what object is in the bounding box.
          score: float, confidence score.
        """
        self.bounding_box = bounding_box
        self.kind = kind
        self.score = score
        self.label = self._LABELS[self.kind]

    def __str__(self):
        return 'kind=%s(%d), score=%f, bbox=%s' % (self._LABELS[self.kind],
                                                   self.kind, self.score,
                                                   str(self.bounding_box))


def _decode_detection_result(logit_scores, box_encodings, anchors,
                             score_threshold, image_size, offset):
    """Decodes result as bounding boxes.
    Args:
      logit_scores: list of scores
      box_encodings: list of bounding boxes
      anchors: list of anchors
      score_threshold: float, bounding box candidates below this threshold will
        be rejected.
      image_size: (width, height)
      offset: (x, y)
    Returns:
      A list of ObjectDetection.Result.
    """
    assert len(box_encodings) == 4 * _NUM_ANCHORS
    assert len(logit_scores) == _NUM_LABELS * _NUM_ANCHORS

    x0, y0 = offset
    width, height = image_size
    objs = []

    score_threshold = max(score_threshold, _MACHINE_EPS)
    logit_score_threshold = math.log(score_threshold / (1 - score_threshold))
    for i in range(_NUM_ANCHORS):
        logits = logit_scores[_NUM_LABELS * i: _NUM_LABELS * (i + 1)]
        max_logit_score = max(logits)
        max_score_index = logits.index(max_logit_score)
        # Skip if max score is below threshold or max score is 'background'.
        if max_score_index == 0 or max_logit_score <= logit_score_threshold:
            continue

        box_encoding = box_encodings[4 * i: 4 * (i + 1)]
        xmin, ymin, xmax, ymax = _decode_box_encoding(box_encoding, anchors[i])
        x = int(x0 + xmin * width)
        y = int(y0 + ymin * height)
        w = int((xmax - xmin) * width)
        h = int((ymax - ymin) * height)
        max_score = 1.0 / (1.0 + math.exp(-max_logit_score))
        objs.append(Object((x, y, w, h), max_score_index, max_score))
    return objs


def _clamp(value):
    """Clamps value to range [0.0, 1.0]."""
    return min(max(0.0, value), 1.0)


def _decode_box_encoding(box_encoding, anchor):
    """Decodes bounding box encoding.
    Args:
      box_encoding: a tuple of 4 floats.
      anchor: a tuple of 4 floats.
    Returns:
      A tuple of 4 floats (xmin, ymin, xmax, ymax), each has range [0.0, 1.0].
    """
    assert len(box_encoding) == 4
    assert len(anchor) == 4
    y_scale = 10.0
    x_scale = 10.0
    height_scale = 5.0
    width_scale = 5.0

    rel_y_translation = box_encoding[0] / y_scale
    rel_x_translation = box_encoding[1] / x_scale
    rel_height_dilation = box_encoding[2] / height_scale
    rel_width_dilation = box_encoding[3] / width_scale

    anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax = anchor
    anchor_ycenter = (anchor_ymax + anchor_ymin) / 2
    anchor_xcenter = (anchor_xmax + anchor_xmin) / 2
    anchor_height = anchor_ymax - anchor_ymin
    anchor_width = anchor_xmax - anchor_xmin

    ycenter = anchor_ycenter + anchor_height * rel_y_translation
    xcenter = anchor_xcenter + anchor_width * rel_x_translation
    height = math.exp(rel_height_dilation) * anchor_height
    width = math.exp(rel_width_dilation) * anchor_width

    # Clamp value to [0.0, 1.0] range, otherwise, part of the bounding box may
    # fall outside of the image.
    xmin = _clamp(xcenter - width / 2)
    ymin = _clamp(ycenter - height / 2)
    xmax = _clamp(xcenter + width / 2)
    ymax = _clamp(ycenter + height / 2)

    return (xmin, ymin, xmax, ymax)


def _area(box):
    _, _, width, height = box
    area = width * height
    assert area >= 0
    return area


def _intersection_area(box1, box2):
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2
    x = max(x1, x2)
    y = max(y1, y2)
    width = max(min(x1 + width1, x2 + width2) - x, 0)
    height = max(min(y1 + height1, y2 + height2) - y, 0)
    area = width * height
    assert area >= 0
    return area


def _overlap_ratio(box1, box2):
    """Computes overlap ratio of two bounding boxes.
    Args:
      box1: (x, y, width, height).
      box2: (x, y, width, height).
    Returns:
      float, represents overlap ratio between given boxes.
    """
    intersection_area = _intersection_area(box1, box2)
    union_area = _area(box1) + _area(box2) - intersection_area
    assert union_area >= 0
    if union_area > 0:
        return float(intersection_area) / float(union_area)
    return 1.0


def _non_maximum_suppression(objs, overlap_threshold=0.5):
    """Runs Non Maximum Suppression.
    Removes candidate that overlaps with existing candidate who has higher
    score.
    Args:
      objs: list of ObjectDetection.Object
      overlap_threshold: float
    Returns:
      A list of ObjectDetection.Object
    """
    objs = sorted(objs, key=lambda x: x.score, reverse=True)
    for i in range(len(objs)):
        if objs[i].score < 0.0:
            continue
        # Suppress any nearby bounding boxes having lower score than boxes[i]
        for j in range(i + 1, len(objs)):
            if objs[j].score < 0.0:
                continue
            if _overlap_ratio(objs[i].bounding_box,
                              objs[j].bounding_box) > overlap_threshold:
                objs[j].score = -1.0  # Suppress box

    return [obj for obj in objs if obj.score >= 0.0]  # Exclude suppressed boxes


def model():
    return ModelDescriptor(
        name='object_detection',
        input_shape=(1, 256, 256, 3),
        input_normalizer=(128.0, 128.0),
        compute_graph=utils.load_compute_graph(_COMPUTE_GRAPH_NAME))


def get_objects(result, score_threshold=0.3, offset=(0, 0)):
    assert len(result.tensors) == 2
    logit_scores = tuple(result.tensors['concat_1'].data)
    box_encodings = tuple(result.tensors['concat'].data)

    size = (result.window.width, result.window.height)
    objs = _decode_detection_result(logit_scores, box_encodings, ANCHORS,
                                    score_threshold, size, offset)
    return _non_maximum_suppression(objs)
