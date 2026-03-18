import cv2
import numpy as np
import math

w, h = 400, 300 
cap = cv2.VideoCapture("feed_clip_0001.mp4")

# Get total number of frames for the seek bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
is_paused = False # Flag for play/pause logic

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

first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while True:
    # 1. Handle Timeline and Playback
    if not is_paused:
        current_pos = cv2.getTrackbarPos("Timeline", "0. Controls")
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        ret, frame = cap.read()
        if not ret:
            cv2.setTrackbarPos("Timeline", "0. Controls", 0)
            continue
        # Update slider to follow video
        cv2.setTrackbarPos("Timeline", "0. Controls", int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    else:
        # If paused, keep reading the same frame to allow slider/filter adjustments
        current_pos = cv2.getTrackbarPos("Timeline", "0. Controls")
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        ret, frame = cap.read()

    # 2. Get Filtering Variables
    b_size = cv2.getTrackbarPos("Blur Size", "0. Controls") | 1 
    t_val = cv2.getTrackbarPos("Threshold", "0. Controls")
    op_k = max(1, cv2.getTrackbarPos("Open Kernel", "0. Controls"))
    op_p = cv2.getTrackbarPos("Open Power", "0. Controls")
    cl_k = max(1, cv2.getTrackbarPos("Close Kernel", "0. Controls"))
    cl_p = cv2.getTrackbarPos("Close Power", "0. Controls")

    first_blur = cv2.GaussianBlur(first_gray, (b_size, b_size), 0)
    
    # --- PROCESSING PIPELINE ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (b_size, b_size), 0)
    diff = cv2.absdiff(first_blur, blur)
    _, thresh = cv2.threshold(diff, t_val, 255, cv2.THRESH_BINARY)
    
    kernel_open = np.ones((op_k, op_k), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=op_p)
    kernel_close = np.ones((cl_k, cl_k), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=cl_p)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            
            # --- CALCULATE ANGLE ---
            angle_rad = 0.5 * math.atan2(2 * M["mu11"], M["mu20"] - M["mu02"])
            angle_deg = int(math.degrees(angle_rad))
            
            # Draw Heading Line
            x2 = int(cx + 40 * math.cos(angle_rad))
            y2 = int(cy + 40 * math.sin(angle_rad))
            cv2.line(frame, (cx, cy), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # --- RESTORED POSITION AND ANGLE INFORMATION ---
            data_text = f"X: {cx} Y: {cy} | Angle: {angle_deg}deg"
            cv2.putText(frame, data_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Status text for Pause
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

    # --- CONTROLS ---
    key = cv2.waitKey(30) & 0xFF
    
    if key == ord('q'): 
        break
    elif key == ord('p'): # 'p' to toggle Pause
        is_paused = not is_paused
    elif key == 32: # Space: restart
        cv2.setTrackbarPos("Timeline", "0. Controls", 0)
    elif key == 81 or key == 2: # Left Arrow
        new_pos = max(0, cv2.getTrackbarPos("Timeline", "0. Controls") - 30)
        cv2.setTrackbarPos("Timeline", "0. Controls", new_pos)
    elif key == 83 or key == 3: # Right Arrow
        new_pos = min(total_frames - 1, cv2.getTrackbarPos("Timeline", "0. Controls") + 30)
        cv2.setTrackbarPos("Timeline", "0. Controls", new_pos)

cap.release()
cv2.destroyAllWindows()