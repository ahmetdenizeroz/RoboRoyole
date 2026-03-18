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

def create_win(name, x, y, w=400, h=300):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.moveWindow(name, x, y)

# --- WINDOW LAYOUT ---
create_win("0. Controls", 0, 0, 300, 850) # Made taller for new checkboxes
create_win("1. Original (Tracking)", 310, 0)
create_win("2. Grayscale", 720, 0)
create_win("3. GrayBlurred", 1130, 0)
create_win("4. Difference", 310, 350)
create_win("5. Threshold", 720, 350)
create_win("6. Opened (Cleaned)", 1130, 350)
create_win("7. Final Mask (Closed)", 720, 700)
create_win("8. Histogram", 1130, 700) 
create_win("9. Zoom View", 310, 700, 400, 300)
create_win("10. ArUco Crop (Gray)", 0, 600, 200, 200) # New window for gray_crop

def nothing(x): pass

cv2.createTrackbar("Timeline", "0. Controls", 0, total_frames - 1, nothing)

# Filtering Trackbars
cv2.createTrackbar("Blur Size", "0. Controls", 7, 51, nothing)
cv2.createTrackbar("Threshold", "0. Controls", 15, 255, nothing)
cv2.createTrackbar("Open Kernel", "0. Controls", 2, 15, nothing)
cv2.createTrackbar("Open Power", "0. Controls", 1, 10, nothing)
cv2.createTrackbar("Close Kernel", "0. Controls", 2, 15, nothing)
cv2.createTrackbar("Close Power", "0. Controls", 2, 10, nothing)
cv2.createTrackbar("Box Width", "0. Controls", 200, 300, nothing)
cv2.createTrackbar("Box Height", "0. Controls", 200, 300, nothing)
cv2.createTrackbar("Show Contour", "0. Controls", 1, 1, nothing)

# --- NEW: WINDOW CHECKBOXES (0=Off, 1=On) ---
cv2.createTrackbar("Show Original", "0. Controls", 1, 1, nothing)
cv2.createTrackbar("Show Gray", "0. Controls", 0, 1, nothing)
cv2.createTrackbar("Show Blur", "0. Controls", 0, 1, nothing)
cv2.createTrackbar("Show Diff", "0. Controls", 1, 1, nothing)
cv2.createTrackbar("Show Thresh", "0. Controls", 0, 1, nothing)
cv2.createTrackbar("Show Opened", "0. Controls", 0, 1, nothing)
cv2.createTrackbar("Show Final", "0. Controls", 1, 1, nothing)
cv2.createTrackbar("Show Hist", "0. Controls", 1, 1, nothing)
cv2.createTrackbar("Show Zoom", "0. Controls", 1, 1, nothing)
cv2.createTrackbar("Show ArUcoCrop", "0. Controls", 0, 1, nothing)

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
    op_k = max(1, cv2.getTrackbarPos("Open Kernel", "0. Controls"))
    op_p = cv2.getTrackbarPos("Open Power", "0. Controls")
    cl_k = max(1, cv2.getTrackbarPos("Close Kernel", "0. Controls"))
    cl_p = cv2.getTrackbarPos("Close Power", "0. Controls")

    first_blur = cv2.GaussianBlur(first_gray, (b_size, b_size), 0)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (b_size, b_size), 0)
    diff = cv2.absdiff(first_blur, blur)
    _, thresh = cv2.threshold(diff, t_val, 255, cv2.THRESH_BINARY)
    
    hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
    hist_canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    log_hist = np.log1p(hist)
    cv2.normalize(log_hist, log_hist, 0, 255, cv2.NORM_MINMAX)
    
    for i in range(1, 256):
        pt1 = (i - 1, 300 - int(log_hist[i-1]))
        pt2 = (i, 300 - int(log_hist[i]))
        cv2.line(hist_canvas, pt1, pt2, (255, 255, 255), 2)
        
    cv2.line(hist_canvas, (t_val, 0), (t_val, 300), (0, 0, 255), 1)

    kernel_open = np.ones((op_k, op_k), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=op_p)
    kernel_close = np.ones((cl_k, cl_k), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=cl_p)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.getTrackbarPos("Show Contour", "0. Controls"):
            cv2.drawContours(frame, [largest_contour], -1, (255, 255, 0), 1)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            box_w = max(10, cv2.getTrackbarPos("Box Width", "0. Controls"))
            box_h = max(10, cv2.getTrackbarPos("Box Height", "0. Controls"))
            
            y1, y2 = np.clip([cy - box_h//2, cy + box_h//2], 0, frame.shape[0])
            x1, x2 = np.clip([cx - box_w//2, cx + box_w//2], 0, frame.shape[1])
            
            if y2 > y1 and x2 > x1:
                bee_crop = frame[y1:y2, x1:x2] 
                gray_crop = cv2.cvtColor(bee_crop, cv2.COLOR_BGR2GRAY)
                #gray_crop = cv2.normalize(gray_crop, None, 0, 255, cv2.NORM_MINMAX)
                corners, ids, rejected = detector.detectMarkers(gray_crop)

                if ids is not None:
                    for i in range(len(ids)):
                        c = corners[i][0]
                        top_left = (int(c[0][0] + x1), int(c[0][1] + y1))
                        cv2.putText(frame, f"ID: {ids[i][0]}", (top_left[0], top_left[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        for j in range(4):
                            pt1 = (int(c[j][0] + x1), int(c[j][1] + y1))
                            pt2 = (int(c[(j+1)%4][0] + x1), int(c[(j+1)%4][1] + y1))
                            cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

                if cv2.getTrackbarPos("Show Zoom", "0. Controls"):
                    scale_factor = 5
                    zoom_view = cv2.resize(bee_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("9. Zoom View", zoom_view)
                
                if cv2.getTrackbarPos("Show ArUcoCrop", "0. Controls"):
                    cv2.imshow("10. ArUco Crop (Gray)", gray_crop)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if is_paused:
        cv2.putText(frame, "PAUSED", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- UPDATED SHOW WINDOWS WITH CHECKBOX LOGIC ---
    if cv2.getTrackbarPos("Show Original", "0. Controls"): cv2.imshow("1. Original (Tracking)", frame)
    if cv2.getTrackbarPos("Show Gray", "0. Controls"):     cv2.imshow("2. Grayscale", gray)
    if cv2.getTrackbarPos("Show Blur", "0. Controls"):     cv2.imshow("3. GrayBlurred", blur)
    if cv2.getTrackbarPos("Show Diff", "0. Controls"):     cv2.imshow("4. Difference", diff)
    if cv2.getTrackbarPos("Show Thresh", "0. Controls"):   cv2.imshow("5. Threshold", thresh)
    if cv2.getTrackbarPos("Show Opened", "0. Controls"):   cv2.imshow("6. Opened (Cleaned)", opened)
    if cv2.getTrackbarPos("Show Final", "0. Controls"):    cv2.imshow("7. Final Mask (Closed)", closed)
    if cv2.getTrackbarPos("Show Hist", "0. Controls"):     cv2.imshow("8. Histogram", hist_canvas)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'): break
    elif key == ord('p'): is_paused = not is_paused
    elif key == 32: cv2.setTrackbarPos("Timeline", "0. Controls", 0)
    elif key in [81, 2]: # Left
        new_pos = max(0, cv2.getTrackbarPos("Timeline", "0. Controls") - 30)
        cv2.setTrackbarPos("Timeline", "0. Controls", new_pos)
    elif key in [83, 3]: # Right
        new_pos = min(total_frames - 1, cv2.getTrackbarPos("Timeline", "0. Controls") + 30)
        cv2.setTrackbarPos("Timeline", "0. Controls", new_pos)

cap.release()
cv2.destroyAllWindows()