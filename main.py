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
        "ratio_pos": (20, 40),
        "direction_pos": (20, 80)
    },
    {
        "name": "Left Eye",
        "iris": LEFT_IRIS,
        "left_corner": 362,
        "right_corner": 263,
        "ratio_pos": None,       # Will be calculated each frame
        "direction_pos": None
    }
]


# -------------------------------
# Helper Functions
# -------------------------------

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


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

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        # -----------------------------------
        # Draw Eye Landmarks
        # -----------------------------------

        for idx in RIGHT_EYE + LEFT_EYE:

            landmark = face.landmark[idx]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.putText(
                frame,
                str(idx),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1
            )

        # -----------------------------------
        # Process Each Eye
        # -----------------------------------

        for eye in EYES:

            iris_points = []

            # Draw Iris Landmarks

            for idx in eye["iris"]:

                landmark = face.landmark[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                iris_points.append((x, y))

                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

                cv2.putText(
                    frame,
                    str(idx),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1
                )

            # -----------------------------------
            # Iris Center
            # -----------------------------------

            center_x = sum(p[0] for p in iris_points) // len(iris_points)
            center_y = sum(p[1] for p in iris_points) // len(iris_points)

            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 255), -1)

            cv2.putText(
                frame,
                "Center",
                (center_x + 8, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1
            )

            # -----------------------------------
            # Eye Corners
            # -----------------------------------

            left_corner = face.landmark[eye["left_corner"]]
            right_corner = face.landmark[eye["right_corner"]]

            left_corner = (
                int(left_corner.x * w),
                int(left_corner.y * h)
            )

            right_corner = (
                int(right_corner.x * w),
                int(right_corner.y * h)
            )

            cv2.circle(frame, left_corner, 4, (255, 255, 255), -1)
            cv2.circle(frame, right_corner, 4, (255, 255, 255), -1)

            # -----------------------------------
            # Ratio Calculation
            # -----------------------------------

            eye_width = euclidean_distance(left_corner, right_corner)

            iris_distance = euclidean_distance(
                left_corner,
                (center_x, center_y)
            )

            ratio = iris_distance / eye_width

            # -----------------------------------
            # Gaze Classification
            # -----------------------------------

            if ratio < 0.40:
                direction = "LEFT"

            elif ratio > 0.60:
                direction = "RIGHT"

            else:
                direction = "CENTER"

            # Left eye text should appear on right side

            if eye["ratio_pos"] is None:

                ratio_pos = (w - 260, 40)
                direction_pos = (w - 260, 80)

            else:

                ratio_pos = eye["ratio_pos"]
                direction_pos = eye["direction_pos"]

            cv2.putText(
                frame,
                f'{eye["name"]}: {ratio:.2f}',
                ratio_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                direction,
                direction_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Eye and Iris Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()