import cv2

# Load the video
cap = cv2.VideoCapture("feed_clip_0001.mp4")

is_frozen = False
frozen_frame = None

print("Controls: [Space] to Freeze/Resume | [Q] to Quit")

while True:
    if not is_frozen:
        ret, frame = cap.read()
        if not ret:
            # Restart video if it ends
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        display_frame = frame
    else:
        # Keep displaying the same frame while frozen
        display_frame = frozen_frame

    # Create the Grayscale version
    gray_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

    # Show Windows
    cv2.imshow("1. Original", display_frame)
    cv2.imshow("2. Grayscale", gray_frame)

    key = cv2.waitKey(30) & 0xFF
    
    if key == ord('q'):
        break
    elif key == 32: # SPACE
        if not is_frozen:
            is_frozen = True
            frozen_frame = frame.copy()
            print("Video Frozen. Press Space to Resume.")
        else:
            is_frozen = False
            print("Video Resumed.")

cap.release()
cv2.destroyAllWindows()