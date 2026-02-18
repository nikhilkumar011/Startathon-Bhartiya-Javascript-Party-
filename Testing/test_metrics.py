##wont run without val folder and data.yaml file and best.pt file in the same directry
##since we are not allowed to upload dataset you wont be able to check the output

from ultralytics import YOLO
import cv2


img_path = "my_test_image.jpg" 
model_path = "best.pt"
data_config = "data.yaml"

def main():
    print(f"Loading model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Running inference on {img_path}")
    results = model(img_path)

    annotated = results[0].plot()
    cv2.imshow("Prediction", annotated)
    print("Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    metrics = model.val(data=data_config, split='val')

    map50 = metrics.seg.map50
    map50_95 = metrics.seg.map

    
    print(f"mAP@50:    {map50:.4f} ({map50*100:.1f}%)")
    print(f"mAP@50-95: {map50_95:.4f} ({map50_95*100:.1f}%)")

if __name__ == "__main__":
    main()