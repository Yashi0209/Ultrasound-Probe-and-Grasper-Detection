# Ultrasound Probe and Grasper Detection

This project uses a YOLOv8 model to detect ultrasound probes and graspers in video footage, drawing bounding boxes and orientation labels on the frames. The processed video is saved to a specified output directory with annotations.

## Features
- Detects ultrasound probes and graspers using a YOLOv8 model (default: fine-tuned YOLOv8s)
- Draws bounding boxes with confidence scores and orientation labels (Horizontal, Vertical, Diagonal)
- Saves the annotated video to a specified output directory
- Supports custom YOLO models via command-line arguments

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Dependencies (install via pip).
- Input Video: A video file (e.g., surgery.mp4) to process
- Model Weights: The default model is a fine-tuned YOLOv8s located at `../model/yolo_v8_n_finetuned/weights/best.pt` relative to the script. You can use your own model by specifying its path.

## Project Structure
ultrasound_probe_detection/                                                                                                                             <br />
├── dataset/                                                                                                                                            <br />
│   └── dataset_yolo_format     # dataset on which YOLO was finetuned                                                                                   <br />
│   |   └── images              # images folder which contains images spilt into training and validation set                                            <br />
│   |   |   └── train                                                                                                                                   <br />
│   |   |   └── val                                                                                                                                     <br />
│   |   └── labels              # labels folder which contains the corresponding labels for the images, split into training and validation set          <br />
│   |   |   └── train                                                                                                                                   <br />
│   |   |   └── val                                                                                                                                     <br /> 
├── src/                                                                                                                                                <br/>    
│   └── video_test.py           # main test script                                                                                                      <br/>
|   └── evaluate.py                                                                                                                                     <br/>
|   └── finetune_YOLOV8.py  <br/>
|   └── m2cai16.yaml       <br/>
├── videos/                     # Directory for input videos (e.g., surgery.mp4) <br/>
|   └── surgery.mp4 <br/>
|   └── video2.mp4 <br/>
|   └── video3.mp4 <br/>
├── model/ <br/>
│   └── yolo_v8_n_finetuned/ <br/>
│       └── weights/ <br/>
│           └── best.pt         # Default fine-tuned YOLOv8s model <br/>
├── results/                    # Output directory for processed videos <br/>
|   └── surgery_result.mp4 <br/>
|   └── screenshots             # folder to keep the screenshots
|   |   └── screenshot.png
└── README.md             <br/>


## Installation
1. Clone or download this repository to your local machine:
git clone <repository-url>
cd inSyteBio_project
2. Install the required Python packages:
pip install -r requirements.txt
3. Place your input video (e.g., surgery.mp4) in the `videos/` directory
4. Ensure the default model (`../model/yolo_v8_n_finetuned/weights/best.pt`) exists, or provide your own model weights

## Usage
Run the script from the `src/` directory using the command-line arguments.

### Basic Command (Using Default Model)
To process a video with the default fine-tuned YOLOv8s model:
cd src
python video_test.py --input-video "../videos/surgery.mp4" --output-file "surgery_result.mp4"

Output will be saved to `src/result/surgery_result.mp4`.

### Custom Output Directory
Specify a different output directory:

python video_test.py --input-video "../videos/surgery.mp4" --output-file "surgery_result.mp4" --output-dir "../results"

Output will be saved to `../results/surgery_result.mp4`.

### Using a Custom Model
To use your own YOLO model instead of the default fine-tuned YOLOv8s:

python video_test.py --input-video "../videos/surgery.mp4" --output-file "surgery_result.mp4" --model-path "path/to/your/model.pt"
Replace `path/to/your/model.pt` with the path to your custom YOLO model weights.

## Command-Line Arguments
| Argument           | Description                                       | Required | Default Value                              |
|--------------------|---------------------------------------------------|----------|--------------------------------------------|
| `--input-video`    | Path to the input video file                      | Yes      | -                                          |
| `--output-file`    | Name of the output video file (e.g., output.mp4)  | Yes      | -                                          |
| `--output-dir`     | Directory to save the output video                | No       | `result` (relative to `src/`)              |
| `--model-path`     | Path to the YOLO model weights                    | No       | `../model/yolo_v8_n_finetuned/weights/best.pt` |

## Example with Absolute Paths
python video_test.py --input-video "absolute_path_of_video.mp4" --output-file "output_file_name" --output-dir "absolute_path_of_output_directory" --model-path "absolute_path_of_model_weight.pt_file"


## Model Details
- **Default Model:** Fine-tuned YOLOv8s (`best.pt`) located at `../model/yolo_v8_n_finetuned/weights/best.pt`
- **Custom Models:** Replace with any YOLO model (e.g., YOLOv8n, YOLOv8m) using `--model-path`. Ensure compatibility with ultralytics library.

## Output
Processed video includes:
- Green bounding boxes with confidence scores and class labels
- Orientation labels (Horizontal, Vertical, Diagonal)
- Saved to specified `--output-dir` with `--output-file` name

## Screenshot
Below is a screenshot of the output video with annotations:
![Output Screenshot](results/screenshots/Screenshot%202025-03-31%20114806.png)
![Output Screenshot](results/screenshots/Screenshot%202025-03-31%20114839.png)

## Troubleshooting
- "Could not open input video": Verify `--input-video` path
- "Could not initialize VideoWriter": Ensure output directory is writable. Try `.avi` format:
--output-file "surgery_result.avi"