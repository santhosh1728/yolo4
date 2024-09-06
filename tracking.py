import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

class ObjectDetection:
    def __init__(self, weights_path="dnn_model/yolov4.weights", cfg_path="dnn_model/yolov4.cfg", classes_path="dnn_model/classes.txt"):
        print("Loading Object Detection")
        print("Running OpenCV DNN with YOLOv4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.classes = self.load_class_names(classes_path)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path):
        try:
            with open(classes_path, "r") as file_object:
                return [class_name.strip() for class_name in file_object.readlines()]
        except FileNotFoundError:
            print(f"Error: Classes file not found at {classes_path}")
            return []

    def detect(self, frame):
        class_ids, scores, boxes = self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
        return class_ids, scores, boxes

class Sort:
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.track_id = 0
        self.track_id_map = {}

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
                self.track_id_map[self.track_id] = self.track_id
                self.track_id += 1
        else:
            iou_matrix = np.zeros((len(self.trackers), len(detections)))
            for t, tracker in enumerate(self.trackers):
                for d, detection in enumerate(detections):
                    iou_matrix[t, d] = self._iou(tracker['bbox'], detection)

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
                    self.track_id_map[self.trackers[t]['id']] = self.track_id_map.get(self.trackers[t]['id'], self.trackers[t]['id'])
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
                self.track_id_map[self.track_id] = self.track_id
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

# Initialize Object Detection and Tracking
od = ObjectDetection()
tracker = Sort()

# Verify the video path
video_path = "Jan 5 SonRise Mom part 1.mp4"
print(f"Attempting to open video file: {video_path}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize counters
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame or end of video")
        break

    count += 1
    print(f"Processing frame {count}")

    # Detect objects on frame
    class_ids, scores, boxes = od.detect(frame)

    # Filter detections for persons only (class ID 0)
    person_boxes = [box for i, box in enumerate(boxes) if class_ids[i] == 0]
    
    # Prepare detections for tracking
    detections = np.array(person_boxes)
    
    if len(detections) > 0:
        trackers = tracker.update(detections)

        # Initialize counters
        person_count = len(trackers)

        for i, track in enumerate(trackers):
            x, y, w, h, track_id = map(int, track)

            # Ensure class_ids has enough elements
            color = (0, 255, 0)  # Default color for persons
            if i < len(class_ids):
                color = od.colors[0]  # Color for class ID 0 (persons)

            # Draw bounding box and ID
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display counts on frame
        cv2.putText(frame, f'Persons: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
