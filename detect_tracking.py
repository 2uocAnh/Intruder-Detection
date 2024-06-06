from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def load_yolo_model(weights_path):
    model = YOLO(weights_path)
    return model

def run_yolo(model, frame):
    results = model(frame, classes=0)
    detections = []
    for result in results:
        bboxes = result.boxes.xyxy.numpy()
        labels = result.boxes.cls.numpy()
        for bbox, label in zip(bboxes, labels):
            if label == 0:  # Assuming '0' is the label for 'person'
                x1, y1, x2, y2 = map(int, bbox)
                bbox = (x1, y1, x2 - x1, y2 - y1)
                detections.append((bbox, label))
    return detections

def initialize_tracker():
    tracker = DeepSort(max_age=30, n_init=3)
    return tracker

def update_tracker(tracker, detections, frame):
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks
