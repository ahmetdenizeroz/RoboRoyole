import cv2
import numpy as np

# 1. Load your video
cap = cv2.VideoCapture("feed_clip_0001.mp4")

# --- ADVANCED SUBTRACTOR SETUP ---
# history: how many frames it remembers
# varThreshold: Mahalanobis distance threshold (higher = less sensitive)
# detectShadows: If True, it marks shadows in gray (127)
subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=15, detectShadows=True)

w, h = 500, 350

# --- WINDOW SETUP ---
def create_win(name, x, y, w=500, h=350):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.moveWindow(name, x, y)

create_win("0. Controls", 0, 0, 300, 450)
create_win("1. Original", w, 0)
create_win("2. MOG2 Raw Mask", 2*w, 0) # This replaces Difference
create_win("3. Threshold", 3*w, 0)
create_win("4. Opened (Cleaned)", w, h)
create_win("5. Closed (Solid)", 2*w, h)

def nothing(x): pass
cv2.createTrackbar("Learning Rate", "0. Controls", 0, 100, nothing) # 0 = Auto, 100 = Instant
cv2.createTrackbar("Threshold", "0. Controls", 15, 255, nothing)
cv2.createTrackbar("Open Kernel", "0. Controls", 2, 15, nothing)
cv2.createTrackbar("Open Power", "0. Controls", 1, 10, nothing)
cv2.createTrackbar("Close Kernel", "0. Controls", 2, 15, nothing)
cv2.createTrackbar("Close Power", "0. Controls", 2, 10, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # --- GET GUI VARIABLES ---
    # Learning rate is usually very small (e.g., 0.001). We divide by 1000.
    lr_val = cv2.getTrackbarPos("Learning Rate", "0. Controls") / 1000.0
    t_val = cv2.getTrackbarPos("Threshold", "0. Controls")
    
    op_k = cv2.getTrackbarPos("Open Kernel", "0. Controls") or 1
    op_p = cv2.getTrackbarPos("Open Power", "0. Controls")
    cl_k = cv2.getTrackbarPos("Close Kernel", "0. Controls") or 1
    cl_p = cv2.getTrackbarPos("Close Power", "0. Controls")

    # --- THE MOG2 STEP ---
    # apply() updates the model and returns the mask in one go
    # Any pixel that is 127 is a "shadow"
    fg_mask = subtractor.apply(frame, learningRate=lr_val)

    # --- THRESHOLD ---
    # We threshold the mask to remove shadows (make them black) 
    # and keep only the white (bee)
    _, thresh = cv2.threshold(fg_mask, t_val, 255, cv2.THRESH_BINARY)

    # --- CLEANING ---
    kernel_open = np.ones((op_k, op_k), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=op_p)

    kernel_close = np.ones((cl_k, cl_k), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=cl_p)

    # --- SHOW RESULTS ---
    cv2.imshow("1. Original", frame)
    cv2.imshow("2. MOG2 Raw Mask", fg_mask)
    cv2.imshow("3. Threshold", thresh)
    cv2.imshow("4. Opened (Cleaned)", opened)
    cv2.imshow("5. Closed (Solid)", closed)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == 32: # Restart
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cap.release()
cv2.destroyAllWindows()