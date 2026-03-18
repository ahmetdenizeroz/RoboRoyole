import cv2
import numpy as np
import math
"""
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
"""
w, h = 400, 300
cap = cv2.VideoCapture("feed_clip_0001.mp4")

# Get total number of frames for the seek bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
is_paused = False

ret, first_frame = cap.read()
if not ret: exit()

def create_win(name, x, y, w=400, h=300):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, w, h)
    cv2.moveWindow(name, x, y)

# --- WINDOW LAYOUT ---
create_win("0. Controls", 0, 0, 300, 550)
create_win("1. Original (Tracking)", 310, 0)
create_win("2. Grayscale", 720, 0)
create_win("3. GrayBlurred", 1130, 0)
create_win("4. Difference", 310, 350)
create_win("5. Threshold", 720, 350)
create_win("6. Opened (Cleaned)", 1130, 350)
create_win("7. Final Mask (Closed)", 720, 700)
create_win("8. Histogram", 1130, 700) # New window for Histogram
create_win("9. Zoom View", 310, 700, 400, 300) # Positioned near the Final Mask
create_win("10. Aruco View", 720, 700, 400, 300)
def nothing(x): pass

# --- TIME CONTROL TRACKBAR ---
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

# --- NEW: ARUCO TRACKBARS ---
cv2.createTrackbar("ArUco Min Size", "0. Controls", 30, 200, nothing)
cv2.createTrackbar("ArUco Max Size", "0. Controls", 400, 1000, nothing)

# --- NEW: ARUCO DICTIONARY SETUP ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

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
    
    # --- UPDATED: LOGARITHMIC HISTOGRAM CALCULATION ---
    hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
    hist_canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Apply Log Scale: log1p(x) is log(1 + x) to avoid log(0) errors
    log_hist = np.log1p(hist)
    
    # Normalize the log values to fit the window height (300 pixels)
    cv2.normalize(log_hist, log_hist, 0, 255, cv2.NORM_MINMAX)
    
    for i in range(1, 256):
        # We use the log_hist values now for drawing
        pt1 = (i - 1, 300 - int(log_hist[i-1]))
        pt2 = (i, 300 - int(log_hist[i]))
        cv2.line(hist_canvas, pt1, pt2, (255, 255, 255), 2)
        
    # Draw a vertical line where the current Threshold trackbar is set
    cv2.line(hist_canvas, (t_val, 0), (t_val, 300), (0, 0, 255), 1)

    kernel_open = np.ones((op_k, op_k), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=op_p)
    kernel_close = np.ones((cl_k, cl_k), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=cl_p)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        show_contour = cv2.getTrackbarPos("Show Contour", "0. Controls")
        if show_contour == 1:
            # -1 draws all points of the contour, (255, 255, 0) is Cyan, thickness is 1
            cv2.drawContours(frame, [largest_contour], -1, (255, 255, 0), 1)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            
            # Get dimensions from trackbars
            box_w = max(10, cv2.getTrackbarPos("Box Width", "0. Controls"))
            box_h = max(10, cv2.getTrackbarPos("Box Height", "0. Controls"))
            
            # Calculate coordinates for cropping (y is rows, x is columns)
            # We use np.clip to ensure coordinates stay within image boundaries [0, h] or [0, w]
            y1, y2 = np.clip([cy - box_h//2, cy + box_h//2], 0, frame.shape[0])
            x1, x2 = np.clip([cx - box_w//2, cx + box_w//2], 0, frame.shape[1])
            
            # --- EXTRACT AND SCALE ROI ---
            if y2 > y1 and x2 > x1:
                bee_crop = frame[y1:y2, x1:x2]
                bee_crop_gray = cv2.cvtColor(bee_crop, cv2.COLOR_BGR2GRAY)
                #bee_crop_norm = cv2.normalize(bee_crop_gray, None, 0, 255, cv2.NORM_MINMAX)
                scale_factor = 3
                zoom_view = cv2.resize(bee_crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                zoom_for_aruco = cv2.resize(bee_crop_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
                
                # --- NEW: ARUCO DETECTION PROCESSING ---
                # Create a color version of the grayscale crop so we can draw green bounding boxes
                aruco_display = cv2.cvtColor(zoom_for_aruco, cv2.COLOR_GRAY2BGR)
                
                # Handle OpenCV version differences for DetectorParameters
                try:
                    parameters = cv2.aruco.DetectorParameters()
                except AttributeError:
                    parameters = cv2.aruco.DetectorParameters_create()
                
                # Apply trackbar values. 
                # Sliders output ints. We divide by 1000/100 to feed the correct decimal percentages to OpenCV
                parameters.minMarkerPerimeterRate = max(1, cv2.getTrackbarPos("ArUco Min Size", "0. Controls")) / 1000.0
                parameters.maxMarkerPerimeterRate = max(1, cv2.getTrackbarPos("ArUco Max Size", "0. Controls")) / 100.0

                corners, ids, rejected = cv2.aruco.detectMarkers(zoom_for_aruco, aruco_dict, parameters=parameters)
                
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(aruco_display, corners, ids)
                    # Draw the ID text manually so it is readable on the scaled crop
                    for i in range(len(ids)):
                        c = corners[i][0]
                        center = (int(c[:, 0].mean()), int(c[:, 1].mean()))
                        cv2.putText(aruco_display, str(ids[i][0]), center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # --- END NEW PROCESSING ---

                cv2.imshow("9. Zoom View", zoom_view)
                cv2.imshow("10. Aruco View", aruco_display) # Changed to show the drawn image

            # Draw the box on the original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if is_paused:
        cv2.putText(frame, "PAUSED", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- SHOW WINDOWS ---
    cv2.imshow("1. Original (Tracking)", frame)
    cv2.imshow("2. Grayscale", gray)
    cv2.imshow("3. GrayBlurred", blur)
    cv2.imshow("4. Difference", diff)
    cv2.imshow("5. Threshold", thresh)
    cv2.imshow("6. Opened (Cleaned)", opened)
    cv2.imshow("7. Final Mask (Closed)", closed)
    cv2.imshow("8. Histogram", hist_canvas) # New Histogram show

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