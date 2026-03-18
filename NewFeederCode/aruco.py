import cv2
import numpy as np
import math

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

w, h = 400, 300 
cap = cv2.VideoCapture("feed_clip_0001.mp4")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
is_paused = False 

ret, first_frame = cap.read()
if not ret: exit()

# --- WINDOW CONFIGURATION DICTIONARY ---
# We store positions and sizes here so we can re-apply them when a window reopens
win_cfg = {
    "1. Original (Tracking)": (310, 0, 400, 300),
    "2. Grayscale": (720, 0, 400, 300),
    "3. GrayBlurred": (1130, 0, 400, 300),
    "4. Difference": (310, 350, 400, 300),
    "5. Threshold": (720, 350, 400, 300),
    "6. Opened (Cleaned)": (1130, 350, 400, 300),
    "7. Final Mask (Closed)": (720, 700, 400, 300),
    "8. Histogram": (1130, 700, 400, 300),
    "9. Zoom View": (310, 700, 400, 300),
    "10. ArUco Crop (Gray)": (0, 600, 200, 200)
}

def manage_window(name, img, state):
    """Handles the actual opening/closing of windows based on trackbar state."""
    if state == 1:
        # If the window doesn't exist, create and position it
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            x, y, ww, wh = win_cfg[name]
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, ww, wh)
            cv2.moveWindow(name, x, y)
        cv2.imshow(name, img)
    else:
        # If the window exists but the state is 0, destroy it
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(name)

# Initial Setup
cv2.namedWindow("0. Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("0. Controls", 300, 850)
cv2.moveWindow("0. Controls", 0, 0)

def nothing(x): pass

cv2.createTrackbar("Timeline", "0. Controls", 0, total_frames - 1, nothing)
cv2.createTrackbar("Blur Size", "0. Controls", 7, 51, nothing)
cv2.createTrackbar("Threshold", "0. Controls", 15, 255, nothing)
cv2.createTrackbar("Open Kernel", "0. Controls", 2, 15, nothing)
cv2.createTrackbar("Open Power", "0. Controls", 1, 10, nothing)
cv2.createTrackbar("Close Kernel", "0. Controls", 2, 15, nothing)
cv2.createTrackbar("Close Power", "0. Controls", 2, 10, nothing)
cv2.createTrackbar("Box Width", "0. Controls", 200, 300, nothing)
cv2.createTrackbar("Box Height", "0. Controls", 200, 300, nothing)
cv2.createTrackbar("Show Contour", "0. Controls", 1, 1, nothing)

# Checkboxes
win_list = ["Original", "Gray", "Blur", "Diff", "Thresh", "Opened", "Final", "Hist", "Zoom", "ArUcoCrop"]
for win in win_list:
    default = 1 if win in ["Original", "Diff", "Hist"] else 0
    cv2.createTrackbar(f"Show {win}", "0. Controls", default, 1, nothing)

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while True:
    if not is_paused:
        current_pos = cv2.getTrackbarPos("Timeline", "0. Controls")
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        ret, frame = cap.read()
        if not ret:
            cv2.setTrackbarPos("Timeline", "0. Controls", 0)
            continue
        cv2.setTrackbarPos("Timeline", "0. Controls", int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    else:
        current_pos = cv2.getTrackbarPos("Timeline", "0. Controls")
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        ret, frame = cap.read()

    b_size = cv2.getTrackbarPos("Blur Size", "0. Controls") | 1 
    t_val = cv2.getTrackbarPos("Threshold", "0. Controls")
    op_k, op_p = max(1, cv2.getTrackbarPos("Open Kernel", "0. Controls")), cv2.getTrackbarPos("Open Power", "0. Controls")
    cl_k, cl_p = max(1, cv2.getTrackbarPos("Close Kernel", "0. Controls")), cv2.getTrackbarPos("Close Power", "0. Controls")

    first_blur = cv2.GaussianBlur(first_gray, (b_size, b_size), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (b_size, b_size), 0)
    diff = cv2.absdiff(first_blur, blur)
    _, thresh = cv2.threshold(diff, t_val, 255, cv2.THRESH_BINARY)
    
    # Histogram
    hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
    hist_canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    log_hist = np.log1p(hist)
    cv2.normalize(log_hist, log_hist, 0, 255, cv2.NORM_MINMAX)
    for i in range(1, 256):
        cv2.line(hist_canvas, (i-1, 300-int(log_hist[i-1])), (i, 300-int(log_hist[i])), (255, 255, 255), 2)
    cv2.line(hist_canvas, (t_val, 0), (t_val, 300), (0, 0, 255), 1)

    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((op_k, op_k), np.uint8), iterations=op_p)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((cl_k, cl_k), np.uint8), iterations=cl_p)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zoom_view = np.zeros((100, 100, 3), np.uint8)
    gray_crop = np.zeros((100, 100), np.uint8)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.getTrackbarPos("Show Contour", "0. Controls"):
            cv2.drawContours(frame, [largest_contour], -1, (255, 255, 0), 1)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            bw, bh = max(10, cv2.getTrackbarPos("Box Width", "0. Controls")), max(10, cv2.getTrackbarPos("Box Height", "0. Controls"))
            y1, y2 = np.clip([cy - bh//2, cy + bh//2], 0, frame.shape[0])
            x1, x2 = np.clip([cx - bw//2, cx + bw//2], 0, frame.shape[1])
            
            if y2 > y1 and x2 > x1:
                bee_crop = frame[y1:y2, x1:x2]
                
                # --- STRATEGY 1: RED CHANNEL EXTRACTION ---
                # Instead of cvtColor, we take index 2 (Red in BGR)
                red_crop = bee_crop[:, :, 2]
                
                # --- STRATEGY 2: CLAHE CONTRAST ENHANCEMENT ---
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_crop = clahe.apply(red_crop)
                
                # --- STRATEGY 3: LANCZOS UPSCALING ---
                # We upscale the ENHANCED crop for better detection
                # Note: Detection on a slightly larger image can help if the marker is tiny
                scale_up = 2 
                gray_crop = cv2.resize(enhanced_crop, None, fx=scale_up, fy=scale_up, 
                                       interpolation=cv2.INTER_LANCZOS4)
                
                # Detect markers on the refined crop
                corners, ids, _ = detector.detectMarkers(gray_crop)
                
                if ids is not None:
                    for i, c in enumerate(corners):
                        c = c[0]
                        # Adjust coordinate translation for the 'scale_up' factor
                        tl = (int((c[0][0] / scale_up) + x1), int((c[0][1] / scale_up) + y1))
                        cv2.putText(frame, f"ID: {ids[i][0]}", (tl[0], tl[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            scale_factor = 5
            zoom_view = cv2.resize(bee_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if is_paused: cv2.putText(frame, "PAUSED", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- WINDOW MANAGEMENT EXECUTION ---
    manage_window("1. Original (Tracking)", frame, cv2.getTrackbarPos("Show Original", "0. Controls"))
    manage_window("2. Grayscale", gray, cv2.getTrackbarPos("Show Gray", "0. Controls"))
    manage_window("3. GrayBlurred", blur, cv2.getTrackbarPos("Show Blur", "0. Controls"))
    manage_window("4. Difference", diff, cv2.getTrackbarPos("Show Diff", "0. Controls"))
    manage_window("5. Threshold", thresh, cv2.getTrackbarPos("Show Thresh", "0. Controls"))
    manage_window("6. Opened (Cleaned)", opened, cv2.getTrackbarPos("Show Opened", "0. Controls"))
    manage_window("7. Final Mask (Closed)", closed, cv2.getTrackbarPos("Show Final", "0. Controls"))
    manage_window("8. Histogram", hist_canvas, cv2.getTrackbarPos("Show Hist", "0. Controls"))
    manage_window("9. Zoom View", zoom_view, cv2.getTrackbarPos("Show Zoom", "0. Controls"))
    manage_window("10. ArUco Crop (Gray)", gray_crop, cv2.getTrackbarPos("Show ArUcoCrop", "0. Controls"))

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'): break
    elif key == ord('p'): is_paused = not is_paused
    elif key == 32: cv2.setTrackbarPos("Timeline", "0. Controls", 0)

cap.release()
cv2.destroyAllWindows()