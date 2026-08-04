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

def calculate_gaze(left_corner, right_corner, upper_eyelid, lower_eyelid, iris_center):
    eye_width = euclidean_distance(left_corner, right_corner)
    eye_height = euclidean_distance(upper_eyelid, lower_eyelid)

    iris_distance = euclidean_distance(left_corner, iris_center)

    ratio = iris_distance / eye_width
    ear = eye_height / eye_width
    if ratio < LEFT_THRESHOLD:direction = "LEFT"
    elif ratio > RIGHT_THRESHOLD:direction = "RIGHT"
    else:direction = "CENTER"


    return ratio, direction, ear

# -------------------------------
# Main Loop
# -------------------------------

while True:
    success, frame = cap.read()
    if not success:
        break
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    EYES[1]["ratio_pos"] = (w - 260, 40)
    EYES[1]["direction_pos"] = (w - 260, 80)
    EYES[1]["ear_pos"] = (w - 260, 120)
    EYES[1]["blink_status_pos"] = (w - 260, 160)
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
                cv2.putText(
                    frame,
                    str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    YELLOW,
                    1
                )

            # -----------------------------------
            # Iris Center
            # -----------------------------------

            iris_center = calculate_iris_center(iris_points)
            cv2.circle(frame, iris_center, 5, PURPLE, -1)
            cv2.putText(
                frame,
                "Center",
                (iris_center[0] + 8, iris_center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                PURPLE,
                1
            )

            # -----------------------------------
            # Eye Corners
            # -----------------------------------

            left_corner = face.landmark[eye["left_corner"]]
            right_corner = face.landmark[eye["right_corner"]]
            upper_eyelid = face.landmark[eye["upper_eyelid"]]
            lower_eyelid = face.landmark[eye["lower_eyelid"]]

            left_corner = landmark_to_pixel(left_corner, w, h)
            right_corner = landmark_to_pixel(right_corner, w, h)
            upper_eyelid = landmark_to_pixel(upper_eyelid, w, h)
            lower_eyelid = landmark_to_pixel(lower_eyelid, w, h)   

            cv2.circle(frame, left_corner, 4, WHITE, -1)
            cv2.circle(frame, right_corner, 4, WHITE, -1)
            cv2.circle(frame, upper_eyelid, 4, WHITE, -1)
            cv2.circle(frame, lower_eyelid, 4, WHITE, -1)

            # -----------------------------------
            # Ratio Calculation
            # -----------------------------------

            ratio,direction,ear=calculate_gaze(left_corner, right_corner, 
                                               upper_eyelid, lower_eyelid, 
                                               iris_center)
            ratio_pos = eye["ratio_pos"]
            direction_pos = eye["direction_pos"]
            ear_pos = eye["ear_pos"]
            blink_status_pos = eye["blink_status_pos"]

            cv2.putText(
                frame,
                f'{eye["name"]}: {ratio:.2f}',
                ratio_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                WHITE,
                2
            )

            cv2.putText(
                frame,
                direction,
                direction_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                GREEN,
                2
            )
            # -----------------------------------
            # Blink Detection
            # -----------------------------------
            if ear < EAR_THRESHOLD:
                blink_status = "BLINKING"
            else:
                blink_status = "NOT BLINKING"
            cv2.putText(
                frame,
                blink_status,
                blink_status_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                GREEN,
                2
            )

    cv2.imshow("Eye and Iris Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()