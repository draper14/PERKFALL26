import cv2
import numpy as np
import time
from flask import Flask, Response

app = Flask(__name__)

# CONFIGURATION
CAMERA_SOURCE = "http://10.186.160.161:8080/video"  # Or 0 for local camera

# Replace your current configuration variables with these:
MOTION_THRESHOLD = 25
MIN_MOTION_PIXELS = 500  # Lowered to ensure we catch the carriage slide
SETTLE_TIME_SEC = 0.25   # Slightly increased to allow mechanical vibration to stop

# [y1, y2, x1, x2] - Make this a narrow horizontal strip covering the active typing line
ROI_CROP = [0.45, 0.58, 0.20, 0.85] 

# Dynamic Red Box tuning (in pixels)
CELL_WIDTH_PX = 60       # Adjust based on how wide a cell appears on camera
CARRIAGE_OFFSET_PX = 15  # Distance from the left edge of the embosser to the braille cell

BRAILLE_DICT = {
    (1,0,0,0,0,0): 'a', (1,1,0,0,0,0): 'b', (1,0,0,1,0,0): 'c',
    (1,0,0,1,1,0): 'd', (1,0,0,0,1,0): 'e', (1,1,0,1,0,0): 'f',
    (1,1,0,1,1,0): 'g', (1,1,0,0,1,0): 'h', (0,1,0,1,0,0): 'i',
    (0,1,0,1,1,0): 'j', (1,0,1,0,0,0): 'k', (1,1,1,0,0,0): 'l',
    (1,0,1,1,0,0): 'm', (1,0,1,1,1,0): 'n', (1,0,1,0,1,0): 'o',
    (1,1,1,1,0,0): 'p', (1,1,1,1,1,0): 'q', (1,1,1,0,1,0): 'r',
    (0,1,1,1,0,0): 's', (0,1,1,1,1,0): 't', (1,0,1,0,0,1): 'u',
    (1,1,1,0,0,1): 'v', (0,1,0,1,1,1): 'w', (1,0,1,1,0,1): 'x',
    (1,0,1,1,1,1): 'y', (1,0,1,0,1,1): 'z', (0,0,0,0,0,0): ' '
}

def parse_braille_cell(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    h, w = thresh.shape
    row_h, col_w = h // 3, w // 2

    dot_positions = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
    binary_matrix = []

    for row, col in dot_positions:
        y1, y2 = row * row_h, (row + 1) * row_h
        x1, x2 = col * col_w, (col + 1) * col_w
        dot_region = thresh[y1:y2, x1:x2]
        ratio = np.sum(dot_region == 255) / dot_region.size
        binary_matrix.append(1 if ratio > 0.18 else 0)

    return tuple(binary_matrix)

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        return

    ret, prev_frame = cap.read()
    if not ret:
        return

    fh, fw, _ = prev_frame.shape
    y1, y2 = int(fh * ROI_CROP[0]), int(fh * ROI_CROP[1])
    x1, x2 = int(fw * ROI_CROP[2]), int(fw * ROI_CROP[3])

    prev_gray = cv2.cvtColor(prev_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    is_moving = False
    last_move_time = time.time()
    last_char = "?"
    last_array = [0, 0, 0, 0, 0, 0]
    
    # Default red box position
    target_x1, target_x2 = x1, x1 + CELL_WIDTH_PX

    while True:
        success, frame = cap.read()
        if not success:
            break

        roi = frame[y1:y2, x1:x2]
        curr_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Motion and Contour Detection
        diff = cv2.absdiff(curr_gray, prev_gray)
        _, thresh_diff = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > MIN_MOTION_PIXELS:
                is_moving = True
                last_move_time = time.time()
                
                # Find the X-coordinate of the moving carriage
                mx, my, mw, mh = cv2.boundingRect(largest_contour)
                
                # Anchor the red box to the left of the carriage movement
                target_x2 = (x1 + mx) - CARRIAGE_OFFSET_PX
                target_x1 = target_x2 - CELL_WIDTH_PX

        # 2. Wait for the carriage to settle, then process the red box
        else:
            if is_moving and (time.time() - last_move_time > SETTLE_TIME_SEC):
                is_moving = False
                
                # Prevent boundary crash if carriage is at extreme left
                target_x1 = max(0, target_x1)
                
                # Crop and parse ONLY the dynamic red box
                red_box_img = frame[y1:y2, target_x1:target_x2]
                if red_box_img.size > 0:
                    braille_tuple = parse_braille_cell(red_box_img)
                    last_array = list(braille_tuple)
                    last_char = BRAILLE_DICT.get(braille_tuple, '?')
                    
                    # Save to log file
                    with open("/home/pi/student_work.txt", "a") as file:
                        file.write(last_char)

        prev_gray = curr_gray.copy()

        # 3. Draw UI Annotations
        status_color = (0, 0, 255) if is_moving else (0, 255, 0)
        
        # Draw the tracking strip (Green)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        # Draw the dynamic braille cell target (Red)
        cv2.rectangle(frame, (target_x1, y1), (target_x2, y2), status_color, 3)
        
        cv2.putText(frame, f"State: {'MOVING' if is_moving else 'SETTLED'}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Array: {last_array}", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Char:  {last_char}", 
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        ret_enc, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)