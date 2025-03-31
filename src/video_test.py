import cv2
import numpy as np
from ultralytics import YOLO


def get_orientation(x1, y1, x2, y2):
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height

    if aspect_ratio > 1.2:
        return "Horizontal"
    elif aspect_ratio < 0.8:
        return "Vertical"
    else:
        return "Diagonal"


def main():
    # Load the trained model
    model = YOLO("runs/detect/m2cai16_tool_old_v33/weights/best.pt")

    # Open a video file (replace with your video path)
    video_path = "surgery.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Draw bounding boxes on the frame
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = det.conf[0]
            cls = int(det.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            orientation = get_orientation(x1, y1, x2, y2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display coordinates and orientation
            info_text = f"({x1}, {y1}) - ({x2}, {y2}) | {orientation}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                info_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        # Display the frame
        cv2.imshow("Ultrasound Probe Detection", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
