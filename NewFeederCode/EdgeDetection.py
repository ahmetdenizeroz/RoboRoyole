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

create_win("0. Controls", 0, 0, 300, 450)
create_win("1. Original", w, 0)
create_win("2. GrayBlurred", 2*w, 0)
create_win("3. Canny Edges", 3*w, 0)
create_win("4. Closed (Solid)", 2*w, h)

def nothing(x): pass

# Canny uses two thresholds to decide what is a "strong" edge and a "weak" edge
cv2.createTrackbar("Threshold 1", "0. Controls", 50, 255, nothing)
cv2.createTrackbar("Threshold 2", "0. Controls", 150, 255, nothing)

# We need a bigger Close Kernel to bridge the gaps between edges
cv2.createTrackbar("Close K", "0. Controls", 5, 30, nothing)
cv2.createTrackbar("Close P", "0. Controls", 2, 10, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1. Preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. Get GUI Variables
    t1 = cv2.getTrackbarPos("Threshold 1", "0. Controls")
    t2 = cv2.getTrackbarPos("Threshold 2", "0. Controls")
    cl_k = cv2.getTrackbarPos("Close K", "0. Controls") or 1
    cl_p = cv2.getTrackbarPos("Close P", "0. Controls")

    # 3. Canny Edge Detection
    edges = cv2.Canny(blur, t1, t2)

    # 4. Fill the outline (Closing)
    # We use a solid kernel to turn the outline into a solid shape
    kernel_cl = np.ones((cl_k, cl_k), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_cl, iterations=cl_p)

    # 5. Show Results
    cv2.imshow("1. Original", frame)
    cv2.imshow("2. GrayBlurred", blur)
    cv2.imshow("3. Canny Edges", edges)
    cv2.imshow("4. Closed (Solid)", closed)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

cap.release()
cv2.destroyAllWindows()