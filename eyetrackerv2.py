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

    },
    {
        "name": "Left Eye",
        "iris": LEFT_IRIS,
        "left_corner": 362,
        "right_corner": 263,
        "upper_eyelid": 386,
        "lower_eyelid": 374,

    }
]


WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
xmin, ymin, xmax, ymax = None, None, None, None

# -------------------------------
# Helper Functions
# -------------------------------

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def landmark_to_pixel(landmark, width, height):
    return (int(landmark.x * width), int(landmark.y * height))


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

        for eye in [EYES[0]]:

            iris_points = []

            # Draw Iris Landmarks

            for idx in eye["iris"]:
                landmark = face.landmark[idx]
                x, y = landmark_to_pixel(landmark, w, h)
                iris_points.append((x, y))
                cv2.circle(frame, (x, y), 3, RED, -1)
                #cv2.putText(frame,str(idx),(x + 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX,0.4,YELLOW,1)


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

            padding_x = 15
            padding_y = 12

            xmin = min(left_corner[0], right_corner[0]) - padding_x
            xmax = max(left_corner[0], right_corner[0]) + padding_x

            ymin = upper_eyelid[1] - padding_y
            ymax = lower_eyelid[1] + padding_y
        
            xmin = max(0, xmin)
            ymin = max(0, ymin)

            xmax = min(w, xmax)
            ymax = min(h, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue

            eye_crop = frame[ymin:ymax, xmin:xmax]
            
            if eye_crop.size == 0:
                continue
            eye_crop = cv2.resize(eye_crop,(128,128))

            cv2.imshow("Eye Crop", eye_crop)


    cv2.imshow("Eye", frame)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()