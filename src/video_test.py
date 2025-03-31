import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse


# Singleton Pattern for Model Loading
class ModelSingleton:
    _instance = None

    def __new__(cls, model_path):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            cls._instance.model = YOLO(model_path)
        return cls._instance

    def get_model(self):
        return self.model


# Strategy Pattern for Orientation Detection
class OrientationStrategy:
    def get_orientation(self, x1, y1, x2, y2):
        raise NotImplementedError("Subclasses must implement this method")


class DefaultOrientationStrategy(OrientationStrategy):
    def get_orientation(self, x1, y1, x2, y2):
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height != 0 else float("inf")
        if aspect_ratio > 1.2:
            return "Horizontal"
        elif aspect_ratio < 0.8:
            return "Vertical"
        else:
            return "Diagonal"


# Factory Pattern for Video Processing
class VideoProcessorFactory:
    def __init__(self, input_path, output_path, model, orientation_strategy):
        self.input_path = input_path
        self.output_path = output_path
        self.model = model
        self.orientation_strategy = orientation_strategy
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open input video: {input_path}")
        self.fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(
            self.output_path, self.fourcc, self.fps, (self.width, self.height)
        )
        if not self.out.isOpened():
            raise ValueError(f"Could not initialize VideoWriter for {self.output_path}")

    def process_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Perform inference
            results = self.model(frame)

            # Draw bounding boxes and annotations on the frame
            for det in results[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0])
                conf = det.conf[0]
                cls = int(det.cls[0])
                label = f"{self.model.names[cls]} {conf:.2f}"
                orientation = self.orientation_strategy.get_orientation(x1, y1, x2, y2)

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

            # Write the annotated frame to the output video
            self.out.write(frame)

            # Display the annotated frame
            cv2.imshow("Ultrasound Probe Detection", frame)
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


# Main Execution Logic
def main(args):
    # Ensure output directory exists
    output_dir = os.path.abspath(args.output_dir)  # Use absolute path for safety
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct full output path
    output_path = os.path.join(output_dir, args.output_file)

    # Load model using Singleton
    model_instance = ModelSingleton(args.model_path)
    model = model_instance.get_model()

    # Set up orientation strategy
    orientation_strategy = DefaultOrientationStrategy()

    # Create and run video processor
    processor = VideoProcessorFactory(
        args.input_video, output_path, model, orientation_strategy
    )
    processor.process_video()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process a video with YOLO model and save output."
    )
    parser.add_argument(
        "--input-video", type=str, required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Name of the output video file (e.g., output.avi)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result",
        help="Directory to save the output video (default: result)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../model/yolo_v8_n_finetuned/weights/best.pt",
        help="Path to the YOLO model weights",
    )

    args = parser.parse_args()
    main(args)
