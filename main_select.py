import cv2
import time
from detect_tracking import load_yolo_model, run_yolo, initialize_tracker, update_tracker
from roi import get_roi, is_intersecting
import os

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    #Lấy frame đầu để chọn ROI
    ret, frame = cap.read()
    roi = get_roi(frame)

    model = load_yolo_model("yolov8n.pt")
    tracker = initialize_tracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Vẽ ROI đã chọn trên các frame từ đấy về sau
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  

        # Sử dụng Yolo để phát hiện
        detections = run_yolo(model, frame)

        # Tracking
        tracks = update_tracker(tracker, detections, frame)

        for i, track in enumerate(tracks):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2] - ltrb[0]), int(ltrb[3] - ltrb[1]))

            if i < len(detections) and detections[i][1] == 0:
                if is_intersecting(bbox, roi):
                    # Create directory for new intruder if it doesn't exist
                    intruder_dir = os.path.join("intruder_list", f"intruder_{track_id}")
                    os.makedirs(intruder_dir, exist_ok=True)

                    # Draw bounding box and track ID on frame before saving
                    color = (0, 255, 0)  # Green color for bounding box
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
                    cv2.putText(frame, f"ID: {track_id}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Draw ROI on frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue color for ROI

                    # Save frame with intruder, bounding box, and track ID
                    frame_filename = os.path.join(intruder_dir, f"intruder_{int(time.time())}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    print(f"Intruder detected and saved as {frame_filename}")

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
