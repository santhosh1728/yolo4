import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.track_id = 0

    def update(self, detections):
        if len(detections) == 0:
            for tracker in self.trackers:
                tracker['age'] += 1
            self.trackers = [tracker for tracker in self.trackers if tracker['age'] <= self.max_age]
            return np.empty((0, 5))

        new_trackers = []
        if len(self.trackers) == 0:
            for detection in detections:
                new_trackers.append({
                    'id': self.track_id,
                    'bbox': detection,
                    'hits': 1,
                    'age': 0
                })
                self.track_id += 1
        else:
            # Calculate IOU
            iou_matrix = np.zeros((len(self.trackers), len(detections)))
            for t, tracker in enumerate(self.trackers):
                for d, detection in enumerate(detections):
                    iou_matrix[t, d] = self._iou(tracker['bbox'], detection)

            # Perform linear assignment
            matched_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = list(zip(matched_indices[0], matched_indices[1]))

            unmatched_trackers = set(range(len(self.trackers)))
            unmatched_detections = set(range(len(detections)))

            for t, d in matched_indices:
                if iou_matrix[t, d] >= self.iou_threshold:
                    new_trackers.append({
                        'id': self.trackers[t]['id'],
                        'bbox': detections[d],
                        'hits': self.trackers[t]['hits'] + 1,
                        'age': 0
                    })
                    unmatched_trackers.discard(t)
                    unmatched_detections.discard(d)

            for t in unmatched_trackers:
                tracker = self.trackers[t]
                tracker['age'] += 1
                if tracker['age'] <= self.max_age:
                    new_trackers.append(tracker)

            for d in unmatched_detections:
                new_trackers.append({
                    'id': self.track_id,
                    'bbox': detections[d],
                    'hits': 1,
                    'age': 0
                })
                self.track_id += 1

        self.trackers = new_trackers
        return np.array([[tracker['bbox'][0], tracker['bbox'][1], tracker['bbox'][2], tracker['bbox'][3], tracker['id']] for tracker in self.trackers])

    def _iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2

        return inter_area / (bbox1_area + bbox2_area - inter_area)
