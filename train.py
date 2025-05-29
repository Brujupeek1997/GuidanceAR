from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("models/yolo11n.pt")

    model.train(
        data="config.yaml",
        epochs=100,
        batch=32,
        imgsz=1020,
        workers=16,
        device=0,
        cache=True,
        rect=True,
        single_cls=False,

        patience=10,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        degrees=0.0,
        translate=0.02,
        scale=0.1,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.1,
        mosaic=0.1,
        mixup=0.0,
        hsv_h=0.001,
        hsv_s=0.3,
        hsv_v=0.2
    )
