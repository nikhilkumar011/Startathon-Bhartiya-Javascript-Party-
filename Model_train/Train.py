from ultralytics import YOLO

def main():
    # Load the Nano model
    model = YOLO('yolov8n-seg.pt') 

    # training starts here
    model.train(
        data='dataset.yaml', 
        epochs=1, 
        imgsz=480,          
        batch=5,            
        device=0,
        project='Trials',
        name='v1_collab_run',
        workers=1,         
        amp=True,
        exist_ok=True
    )

if __name__ == '__main__':

    main()
