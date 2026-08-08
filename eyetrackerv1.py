import cv2
import mediapipe as mp

# -------------------------------
# MediaPipe Initialization
# -------------------------------

mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# -------------------------------
# Landmark IDs
# -------------------------------

RIGHT_EYE = [33, 133, 159, 145, 153, 154, 155]
LEFT_EYE = [362, 263, 386, 374, 380, 381, 382]

RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]

# -------------------------------
# Eye Configuration
# -------------------------------

EYES = [
    {
        "name": "Right Eye",
        "iris": RIGHT_IRIS,
        "left_corner": 33,
        "right_corner": 133,
        "upper_eyelid": 159,
        "lower_eyelid": 145,
        "top_iris": 470,
        "bottom_iris": 472,
        "ratio_pos": (20, 40),
        "direction_pos": (20, 80),
        "ear_pos": (20, 120),  # Position for EAR display
        "blink_status_pos": (20, 160)  # Position for blink status display
    },
    {
        "name": "Left Eye",
        "iris": LEFT_IRIS,
        "left_corner": 362,
        "right_corner": 263,
        "upper_eyelid": 386,
        "lower_eyelid": 374,
        "top_iris": 475,
        "bottom_iris": 477,
        "ratio_pos": None,       # Will be calculated each frame
        "direction_pos": None,   # Will be calculated each frame
        "ear_pos": None,          # Will be calculated each frame
        "blink_status_pos": None  # Will be calculated each frame
    }
]

EAR_THRESHOLD = 0.20
LEFT_THRESHOLD = 0.40
RIGHT_THRESHOLD = 0.60

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)

BOX_WIDTH = 400
BOX_HEIGHT = 250
BOX_X = 100
BOX_Y = 100

SMOOTHING_WINDOW = 5
horizontal_history = []
vertical_history = []

left_cal=right_cal=None
up_cal=down_cal=None
calibrated = False

# -------------------------------
# Helper Functions
# -------------------------------

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def landmark_to_pixel(landmark, width, height):
    return (int(landmark.x * width), int(landmark.y * height))

def calculate_iris_center(iris_points):
    x_coords = [point[0] for point in iris_points]
    y_coords = [point[1] for point in iris_points]

    center_x = sum(x_coords) // len(x_coords)
    center_y = sum(y_coords) // len(y_coords)

    return (center_x, center_y)

def calculate_gaze(left_corner, right_corner, upper_eyelid, lower_eyelid, upper_iris, lower_iris, iris_center):
    eye_width = euclidean_distance(left_corner, right_corner)
    eye_height = euclidean_distance(upper_eyelid, lower_eyelid)

    iris_horizontal_distance = iris_center[0] - left_corner[0]
    horizontal_ratio = iris_horizontal_distance / eye_width

    #iris_vertical_distance = iris_center[1] - upper_eyelid[1]
    #vertical_offset= upper_eyelid[1] - iris_center[1]
    #vertical_ratio = vertical_offset / eye_height
    top_gap = upper_iris[1] - upper_eyelid[1]
    bottom_gap = lower_eyelid[1] - lower_iris[1]
    vertical_ratio = bottom_gap / (top_gap + bottom_gap) if (top_gap + bottom_gap) != 0 else 0
    
 
    ear = eye_height / eye_width

    if horizontal_ratio < LEFT_THRESHOLD:direction = "LEFT"
    elif horizontal_ratio > RIGHT_THRESHOLD:direction = "RIGHT"
    else:direction = "CENTER"

    if eye_width < 1 or eye_height < 1:
        return None, None, None, None
    return horizontal_ratio, vertical_ratio, direction, ear

# -------------------------------
# Main Loop
# -------------------------------

while True:
    key=cv2.waitKey(1) & 0xFF
    success, frame = cap.read()
    if not success:
        break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if EYES[1]["ratio_pos"] is None:
        EYES[1]["ratio_pos"] = (w - 260, 40)
        EYES[1]["direction_pos"] = (w - 260, 80)
        EYES[1]["ear_pos"] = (w - 260, 120)
        EYES[1]["blink_status_pos"] = (w - 260, 160)

    raw_right_horizontal_ratio = None
    raw_left_horizontal_ratio = None

    raw_right_vertical_ratio = None
    raw_left_vertical_ratio = None

    if results.multi_face_landmarks:
    
        face = results.multi_face_landmarks[0]

        # -----------------------------------
        # Draw Eye Landmarks
        # -----------------------------------

        #for idx in RIGHT_EYE + LEFT_EYE:
            #landmark = face.landmark[idx]
            #x, y = landmark_to_pixel(landmark, w, h)
            #cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            #cv2.putText(frame,str(idx),(x + 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 255),1)

        # -----------------------------------
        # Process Each Eye
        # -----------------------------------

        for eye in EYES:

            iris_points = []

            # Draw Iris Landmarks

            for idx in eye["iris"]:
                landmark = face.landmark[idx]
                x, y = landmark_to_pixel(landmark, w, h)
                iris_points.append((x, y))
                cv2.circle(frame, (x, y), 3, RED, -1)
                #cv2.putText(frame,str(idx),(x + 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX,0.4,YELLOW,1)

            # -----------------------------------
            # Iris Center
            # -----------------------------------

            iris_center = calculate_iris_center(iris_points)
            cv2.circle(frame, iris_center, 5, PURPLE, -1)
            cv2.putText(frame,"Center",(iris_center[0] + 8, iris_center[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,PURPLE,1)

            # -----------------------------------
            # Eye Corners
            # -----------------------------------

            left_corner = face.landmark[eye["left_corner"]]
            right_corner = face.landmark[eye["right_corner"]]
            upper_eyelid = face.landmark[eye["upper_eyelid"]]
            lower_eyelid = face.landmark[eye["lower_eyelid"]]
            upper_iris = face.landmark[eye["top_iris"]]
            lower_iris = face.landmark[eye["bottom_iris"]]

            left_corner = landmark_to_pixel(left_corner, w, h)
            right_corner = landmark_to_pixel(right_corner, w, h)
            upper_eyelid = landmark_to_pixel(upper_eyelid, w, h)
            lower_eyelid = landmark_to_pixel(lower_eyelid, w, h)  
            upper_iris = landmark_to_pixel(upper_iris, w, h)
            lower_iris = landmark_to_pixel(lower_iris, w, h) 

            cv2.circle(frame, left_corner, 4, WHITE, -1)
            cv2.circle(frame, right_corner, 4, WHITE, -1)
            cv2.circle(frame, upper_eyelid, 4, WHITE, -1)
            cv2.circle(frame, lower_eyelid, 4, WHITE, -1)
            cv2.circle(frame, upper_iris, 5, (255,255,0), -1)
            cv2.circle(frame, lower_iris, 5, (255,255,0), -1)

            # -----------------------------------
            # Ratio Calculation
            # -----------------------------------

            raw_horizontal_ratio,raw_vertical_ratio,direction,ear=calculate_gaze(left_corner, right_corner,upper_eyelid, lower_eyelid, upper_iris, lower_iris, iris_center)
            ratio_pos = eye["ratio_pos"]
            direction_pos = eye["direction_pos"]
            ear_pos = eye["ear_pos"]
            blink_status_pos = eye["blink_status_pos"]
            if eye["name"] == "Right Eye":
                raw_right_horizontal_ratio = raw_horizontal_ratio
                raw_right_vertical_ratio = raw_vertical_ratio
            else:
                raw_left_horizontal_ratio = raw_horizontal_ratio
                raw_left_vertical_ratio = raw_vertical_ratio

            cv2.putText(frame,direction,direction_pos,cv2.FONT_HERSHEY_SIMPLEX,1,GREEN,2)
            # -----------------------------------
            # Blink Detection
            # -----------------------------------
            if ear < EAR_THRESHOLD:
                blink_status = "BLINKING"
            else:
                blink_status = "NOT BLINKING"
            cv2.putText(frame,blink_status,blink_status_pos,cv2.FONT_HERSHEY_SIMPLEX,0.6,GREEN,2)

    if raw_right_horizontal_ratio is not None and raw_left_horizontal_ratio is not None:
        average_horizontal_ratio = (raw_right_horizontal_ratio + raw_left_horizontal_ratio) / 2
        average_vertical_ratio = (raw_right_vertical_ratio + raw_left_vertical_ratio) / 2
        cv2.putText(frame,f"H: {average_horizontal_ratio:.2f}",(20, h - 45),cv2.FONT_HERSHEY_SIMPLEX,0.6,WHITE,2)
        cv2.putText(frame,f"V: {average_vertical_ratio:.2f}",(20, h - 20),cv2.FONT_HERSHEY_SIMPLEX,0.6,WHITE,2)
        cv2.rectangle(frame,(BOX_X, BOX_Y),(BOX_X + BOX_WIDTH, BOX_Y + BOX_HEIGHT),WHITE,2)

        horizontal_history.append(average_horizontal_ratio)
        if len(horizontal_history) > SMOOTHING_WINDOW:
            horizontal_history.pop(0)
        smoothed_horizontal_ratio = sum(horizontal_history) / len(horizontal_history)

        vertical_history.append(average_vertical_ratio)
        if len(vertical_history) > SMOOTHING_WINDOW:
            vertical_history.pop(0)
        smoothed_vertical_ratio = sum(vertical_history) / len(vertical_history)

    if horizontal_history and vertical_history:
        smoothed_horizontal_ratio = max(0.0, min(smoothed_horizontal_ratio, 1.0))
        smoothed_vertical_ratio = max(0.0, min(smoothed_vertical_ratio, 1.0))
        #dot_x = BOX_X + int(smoothed_horizontal_ratio * BOX_WIDTH)
        #dot_y = BOX_Y + int(smoothed_vertical_ratio * BOX_HEIGHT)
        if key== ord('.'): 
            calibrated = False
            left_cal=right_cal=None
            up_cal=down_cal=None
        status = lambda v: "✓" if v is not None else "✗"
        cv2.putText(frame,f"L {status(left_cal)}  R {status(right_cal)}  U {status(up_cal)}  D {status(down_cal)}",(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,GREEN,2)
        if not calibrated:
            cv2.putText(frame,"Calibration: Press L,R,U,D while looking in those directions",(20, 30),cv2.FONT_HERSHEY_SIMPLEX,0.6,GREEN,2)
            if key in (ord('l'), ord('L')):
                left_cal = smoothed_horizontal_ratio
            if key in (ord('r'), ord('R')):
                right_cal = smoothed_horizontal_ratio
            if key in (ord('u'), ord('U')):
                up_cal = smoothed_vertical_ratio
            if key in (ord('d'), ord('D')):
                down_cal = smoothed_vertical_ratio
            if None not in (left_cal, right_cal, up_cal, down_cal):
                calibrated = True     
        


        if None not in (left_cal, right_cal) and abs(right_cal-left_cal) > 1e-6:
            normalized_x = (smoothed_horizontal_ratio - left_cal) / (right_cal - left_cal)
            cv2.putText(frame,f"Calibrated H: {left_cal:.2f} - {right_cal:.2f}",(20, h - 70),cv2.FONT_HERSHEY_SIMPLEX,0.6,WHITE,2)
        else:
            normalized_x = 0.0

        if None not in (up_cal, down_cal) and abs(down_cal-up_cal) > 1e-6:
            normalized_y = (smoothed_vertical_ratio - up_cal) / (down_cal - up_cal)
            cv2.putText(frame,f"Calibrated V: {up_cal:.2f} - {down_cal:.2f}",(20, h - 95),cv2.FONT_HERSHEY_SIMPLEX,0.6,WHITE,2)
        else:
            normalized_y = 0.0

        normalized_x = max(0.0, min(normalized_x, 1.0))
        normalized_y = max(0.0, min(normalized_y, 1.0)) 
        dot_x = BOX_X + int(normalized_x * BOX_WIDTH)
        dot_y = BOX_Y + int(normalized_y * BOX_HEIGHT)
        cv2.circle(frame,(dot_x, dot_y),8,GREEN,-1)


    cv2.imshow("Eye and Iris Landmarks", frame)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()