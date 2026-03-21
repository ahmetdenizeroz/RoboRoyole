import cv2
import time
from ultralytics import YOLO

# 1. Load the 'Nano' model (fastest for CPU)
# This will download the 6MB 'yolov8n.pt' file automatically
model = YOLO("yolov8n.pt") 

# 2. Open your video file
video_path = "test.mp4" # <--- Update this path!
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Running on CPU. Expect lower FPS than the 3060...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- THE CPU CORE ---
    # device='cpu' forces the math onto your AMD processor
    # stream=True is a memory-saving trick for processing videos
    results = model.predict(frame, device='cpu', conf=0.25, stream=True)

    for r in results:
        # Calculate CPU Inference Speed
        inference_time = r.speed['inference']
        fps = 1000 / inference_time if inference_time > 0 else 0

        # Plot the boxes (it will likely call your bee a 'bird')
        annotated_frame = r.plot()

        # Display the FPS
        cv2.putText(annotated_frame, f"CPU FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2) # Red for CPU

        cv2.imshow("YOLOv8 Bee Detection - CPU Mode", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()