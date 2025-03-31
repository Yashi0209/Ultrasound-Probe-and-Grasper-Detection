import torch
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback


def on_fit_epoch_end(trainer):
    """Log metrics at the end of each epoch"""
    wandb.log({**trainer.lr, **trainer.metrics})


def main():
    # Step 1: Check if GPU is available
    # wandb.login('f1c13300f3ca4d9a8ee070013702b288cedeb55e')
    run = wandb.init(project="yolo", name="m2cai16_tool_sgd2", job_type="training")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Step 2: Load the YOLOv8 model and move it to the GPU (if available)
    model = YOLO("yolov8n.pt").to(device)
    # model = YOLO(
    #     "E:/Ayushi/object_detection/runs/detect/m2cai16_tool_old_v2.03/weights/best.pt"
    # ).to(device)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # Step 3: Fine-tune the model with explicit GPU usage
    model.train(
        data="m2cai16.yaml",  # Path to your dataset YAML file
        epochs=35,  # Number of training epochs
        batch=16,  # Batch size
        imgsz=640,  # Image size
        # optimizer='AdamW',          # Use SGD optimizer
        # lr0=0.01,                 # Initial learning rate
        # momentum=0.937,           # Momentum for SGD
        # weight_decay=0.0005,      # Weight decay for regularization
        # cos_lr=True,              # Use cosine annealing scheduler
        # warmup_epochs=3,          # Warmup for 3 epochs
        patience=5,  # Early stopping patience
        name="m2cai16_tool_old_v3",  # Experiment name
        verbose=True,
    )

    print("Training completed.")
    wandb.finish()


if __name__ == "__main__":
    main()
