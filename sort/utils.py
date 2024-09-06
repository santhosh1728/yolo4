import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    xx1, yy1, xx2, yy2 = bbox2

    inter_area = max(0, min(x2, xx2) - max(x1, xx1)) * max(0, min(y2, yy2) - max(y1, yy1))
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (xx2 - xx1) * (yy2 - yy1)
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area

def update_tracks(detections, trackers, frame_count):
    if len(trackers) == 0:
        trackers = [[detection[0], detection[1], detection[2], detection[3], -1] for detection in detections]
        return trackers

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for i, detection in enumerate(detections):
        for j, tracker in enumerate(trackers):
            iou_matrix[i, j] = iou(detection, tracker[:4])

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    updated_trackers = []
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= 0.3:
            updated_trackers.append(detections[i] + [j])
        else:
            updated_trackers.append([0, 0, 0, 0, -1])

    trackers = updated_trackers
    return trackers
