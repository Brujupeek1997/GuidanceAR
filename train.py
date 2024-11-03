from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("models/yolo11x.pt")

    model.train(data="config.yaml", imgsz=640, epochs=300, batch=16, workers=8, device=0, optimizer="AdamW", lr0=0.001,
        lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3, warmup_momentum=0.8, warmup_bias_lr=0.1,
        box=7.5, cls=0.5, dfl=1.5, label_smoothing=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.1, save=True, plots=True, patience=50, overlap_mask=False,
        mask_ratio=2)
