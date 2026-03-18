import cv2
import numpy as np

# 1. Load your video
cap = cv2.VideoCapture("feed_clip_0001.mp4")

w, h = 500, 350

# --- WINDOW SETUP ---
def create_win(name, x, y, w=500, h=350):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.moveWindow(name, x, y)

create_win("0. Controls", 0, 0, 300, 550)
create_win("1. Original", w, 0)
create_win("2. HSV Space", 2*w, 0)
create_win("3. Color Mask", 3*w, 0)
create_win("4. Cleaned (Opening)", w, h)
create_win("5. Final (Closing)", 2*w, h)

def nothing(x): pass

# HSV range sliders
cv2.createTrackbar("L-H", "0. Controls", 0, 179, nothing) # Lower Hue
cv2.createTrackbar("L-S", "0. Controls", 0, 255, nothing) # Lower Sat
cv2.createTrackbar("L-V", "0. Controls", 0, 255, nothing) # Lower Val
cv2.createTrackbar("U-H", "0. Controls", 179, 179, nothing) # Upper Hue
cv2.createTrackbar("U-S", "0. Controls", 255, 255, nothing) # Upper Sat
cv2.createTrackbar("U-V", "0. Controls", 255, 255, nothing) # Upper Val

# Morphology sliders
cv2.createTrackbar("Open K", "0. Controls", 2, 15, nothing)
cv2.createTrackbar("Close K", "0. Controls", 2, 15, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1. Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Get Range Values from GUI
    l_h = cv2.getTrackbarPos("L-H", "0. Controls")
    l_s = cv2.getTrackbarPos("L-S", "0. Controls")
    l_v = cv2.getTrackbarPos("L-V", "0. Controls")
    u_h = cv2.getTrackbarPos("U-H", "0. Controls")
    u_s = cv2.getTrackbarPos("U-S", "0. Controls")
    u_v = cv2.getTrackbarPos("U-V", "0. Controls")

    lower_color = np.array([l_h, l_s, l_v])
    upper_color = np.array([u_h, u_s, u_v])

    # 3. Create Mask (The "Threshold" step)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 4. Cleaning
    op_k = cv2.getTrackbarPos("Open K", "0. Controls") or 1
    cl_k = cv2.getTrackbarPos("Close K", "0. Controls") or 1
    
    kernel_op = np.ones((op_k, op_k), np.uint8)
    kernel_cl = np.ones((cl_k, cl_k), np.uint8)
    
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_op)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_cl)

    # 5. Show Results
    cv2.imshow("1. Original", frame)
    cv2.imshow("2. HSV Space", hsv) # Looks weird/trippy, that's normal!
    cv2.imshow("3. Color Mask", mask)
    cv2.imshow("4. Cleaned (Opening)", opened)
    cv2.imshow("5. Final (Closing)", closed)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cap.release()
cv2.destroyAllWindows()