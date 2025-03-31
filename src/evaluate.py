from ultralytics import YOLO


def main():

    # Load the trained model (use the best weights from training)
    model = YOLO("runs/detect/m2cai16_tool_old_v3/weights/best.pt")

    # Evaluate on the validation set
    val_results = model.val(data="m2cai16.yaml", split="val")
    print("Validation Results:", val_results)

    # Evaluate on the test set
    # test_results = model.val(data="m2cai16.yaml", split="test")
    # print("Test Results:", test_results)


if __name__ == "__main__":
    main()
