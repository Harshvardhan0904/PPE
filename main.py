import cv2
import mediapipe as mp
import math
import time 
import os 

# Initialize MediaPipe Pose model
custom  = "violation people"
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Function to calculate distance from point to line
def distance_from_point_to_line(point, line_start, line_end):
    x1, y1 = line_start
    x2, y2 = line_end
    x0, y0 = point

    AB = (x2 - x1, y2 - y1)
    AC = (x0 - x1, y0 - y1)
    dot_product = AB[0] * AC[0] + AB[1] * AC[1]
    AB_squared_length = AB[0] * AB[0] + AB[1] * AB[1]
    t = max(0, min(1, dot_product / AB_squared_length))
    closest_point = (x1 + t * AB[0], y1 + t * AB[1])
    distance = math.sqrt((x0 - closest_point[0]) ** 2 + (y0 - closest_point[1]) ** 2)
    return distance, closest_point

# Function to draw a transparent green area around the specified line
def draw_transparent_green_area(frame, start_point, angle_degrees, line_length, thickness=15):
    overlay = frame.copy()
    output = frame.copy()
    angle_radians = math.radians(angle_degrees)
    end_point = (
        int(start_point[0] + line_length * math.cos(angle_radians)),
        int(start_point[1] + line_length * math.sin(angle_radians))
    )
    cv2.line(overlay, start_point, end_point, (0, 255, 0), thickness)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output, end_point
def crop_center(image, crop_x, crop_y):
    y, x, _ = image.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return image[start_y:start_y+crop_y, start_x:start_x+crop_x]
# Video capture setup
video_path = "amogh4.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Failed to open video file.")
    exit()

# Line parameters
start_point = (0, 380) # Adjust as necessary
angle_degrees = 372
initial_line_length = 380
line_length = initial_line_length

safe_count = 0
violation_count = 0
violation_start_time = None
screenshot_frame = None
is_violation = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1000, 800))
    frame_with_line, line_end_point = draw_transparent_green_area(frame, start_point, angle_degrees, line_length)
    frame_rgb = cv2.cvtColor(frame_with_line, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    safe = False

    if results.pose_landmarks:
        min_distance = float('inf')
        landmarks = results.pose_landmarks.landmark
        keypoints = [
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_THUMB,
            mp_pose.PoseLandmark.RIGHT_THUMB,
            mp_pose.PoseLandmark.LEFT_PINKY,
            mp_pose.PoseLandmark.RIGHT_PINKY,
            mp_pose.PoseLandmark.LEFT_INDEX,
            mp_pose.PoseLandmark.RIGHT_INDEX
        ]

        for keypoint in keypoints:
            x = int(landmarks[keypoint.value].x * frame.shape[1])
            y = int(landmarks[keypoint.value].y * frame.shape[0])
            distance_to_line, _ = distance_from_point_to_line((x, y), start_point, line_end_point)
            if distance_to_line < min_distance:
                min_distance = distance_to_line

            cv2.circle(frame_with_line, (x, y), 3, (0, 255, 0) if keypoint.value % 2 == 0 else (0, 0, 255), cv2.FILLED)

        if min_distance < 20:
            safe = True

    if safe:
        safe_count += 1
        violation_start_time = None
        screenshot_frame = None 
        is_violation = False
    else:
        violation_count += 1
        if violation_start_time is None:
            violation_start_time = time.time()
        elif time.time() - violation_start_time > 3 and screenshot_frame is None:
            screenshot_frame = frame_with_line.copy()
            is_violation = True
    

    current_status = "" if safe_count > violation_count else "Violation Detected"
    cv2.putText(frame_with_line, current_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0) if current_status == "Safe" else (0, 0, 255), 2, cv2.LINE_AA)
    

    if screenshot_frame is not None:
        # Resize the screenshot frame for display in the top right corner
        screenshot_resized = cv2.resize(screenshot_frame, (frame.shape[1] // 3, frame.shape[0] // 3))
        h, w = screenshot_resized.shape[:2]
        
        # Place the screenshot in the top right corner of the frame
        frame_with_line[0:h, frame.shape[1]-w:frame.shape[1]] = screenshot_resized
        cv2.imwrite(os.path.join(custom, "violation_screenshot.jpg"), screenshot_frame)


    cv2.imshow('Live Video with Transparent Green Area and Pose Landmarks Detection', frame_with_line)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("safe :", safe_count)
print("violatiom :", violation_count)
cap.release()
cv2.destroyAllWindows()