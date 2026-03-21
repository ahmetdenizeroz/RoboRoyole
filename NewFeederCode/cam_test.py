import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Use 1 for Logitech
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# Try a low resolution to see if bandwidth is the issue
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
cap.set(cv2.CAP_PROP_FPS, 30)

# FORCE Exposure to a very low number to ensure sensor speed
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
cap.set(cv2.CAP_PROP_EXPOSURE, -7) 

while True:
    t1 = time.time()
    ret, frame = cap.read()
    if not ret: break
    
    fps = 1 / (time.time() - t1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 0, 0.7, (0,255,0), 2)
    cv2.imshow("TEST", frame)
    if cv2.waitKey(1) == ord('q'): break
cap.release()