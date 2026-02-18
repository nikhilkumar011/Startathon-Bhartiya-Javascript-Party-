from ultralytics import YOLO

def main():
    # Load the Nano model
    model = YOLO('yolov8n-seg.pt') 

    # Start Low-Memory Training
    model.train(
        data='dataset.yaml', 
        epochs=1, 
        imgsz=480,          # Reduced from 640 -> 480 (Massive memory saver)
        batch=5,            # Reduced from 8 -> 4 (Fits in 6GB VRAM)
        device=0,
        project='Trials',
        name='v1_collab_run',
        workers=1,          # Keep at 0 for stability
        amp=True,
        exist_ok=True
    )

if __name__ == '__main__':
    main()