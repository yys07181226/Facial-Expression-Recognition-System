import cv2
import numpy as np
from ultralytics import YOLO
import time

# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "C:/vscodedev/pydev/venv/Lib/site-packages/PyQt5/Qt/plugins/platforms"


def run_emotion_recognition(
    model_path: str,  # Path to the trained model weights
    class_names: list,  # List of emotion class names
    device: str = "0",  # Running device, "0" for GPU, "cpu" for CPU
    cam_id: int = 0,  # Camera ID, typically 0 is the default camera
    imgsz: int = 640  # Inference image size, keep consistent with training
):
    # Load model
    print(f"loading model: {model_path}")
    model = YOLO(model_path)
    model.to(device)  # Load model to specified device
    
    # Open camera
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("cannot open camera")
        return
    
    print("### START ###\n press 'q' to quit")
    
    while True:
        # Read a frame
        ret, frame = cap.read() # ret is a BOOLEAN indicating if frame is read correctly
        if not ret:
            Warning("cannot read frame from camera")
            break
        
        # Mirror flip (optional, makes the display more natural)
        frame = cv2.flip(frame, 1)
        
        # Model inference
        start_time = time.time()
        results = model(frame, imgsz=imgsz, device=device)
        infer_time = (time.time() - start_time) * 1000  # Inference time (milliseconds)
        
        # Process recognition results
        annotated_frame = results[0].plot()  # Automatically draw bounding boxes and classes
        
        # Add additional information to the frame, including inference time and FPS
        cv2.putText(
            annotated_frame,
            f"Infer Time: {infer_time:.2f} ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            annotated_frame,
            f"FPS: {int(1000/infer_time) if infer_time > 0 else 0}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Display results
        cv2.imshow("Emotion Recognition", annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("### END ###")

if __name__ == "__main__":
    # modidied parameters
    MODEL_PATH = "C:/Users/yangyousen/Desktop/my project/Facial Expression Recognition System/best.pt"
    CLASS_NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
    Device = "cpu" # "0" for GPU, "cpu" for CPU
    CAM_ID = 0 # set camera id
    IMGSZ = 320 
    
    # run main function
    run_emotion_recognition(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        device=Device,
        cam_id=CAM_ID,
        imgsz=IMGSZ
    ) 