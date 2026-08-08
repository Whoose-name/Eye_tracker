import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import random
import time

# -----------------------
# Dataset
# -----------------------

SAVE_DIR = "dataset/images"
LABEL_FILE = "dataset/labels.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.exists(LABEL_FILE):
    with open(LABEL_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "x", "y"])

image_counter = len(os.listdir(SAVE_DIR))

# -----------------------
# MediaPipe
# -----------------------

mp_face = mp.solutions.face_mesh

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# -----------------------
# Right Eye
# -----------------------

LEFT_CORNER = 33
RIGHT_CORNER = 133
UPPER = 159
LOWER = 145

PADDING_X = 15
PADDING_Y = 12

# -----------------------
# Fullscreen target window
# -----------------------

cv2.namedWindow("Target", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Target",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

screen = np.zeros((1080, 1920, 3), dtype=np.uint8)

screen_h, screen_w = screen.shape[:2]

margin = 100

target_x = random.randint(margin, screen_w - margin)
target_y = random.randint(margin, screen_h - margin)

captures_per_target = 10
captured = 0

settle_time = 0.5
last_move = time.time()

# -----------------------
# Helper
# -----------------------

def pixel(lm, w, h):
    return int(lm.x*w), int(lm.y*h)

# -----------------------
# Main Loop
# -----------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        left = pixel(face.landmark[LEFT_CORNER], w, h)
        right = pixel(face.landmark[RIGHT_CORNER], w, h)
        upper = pixel(face.landmark[UPPER], w, h)
        lower = pixel(face.landmark[LOWER], w, h)

        xmin = min(left[0], right[0]) - PADDING_X
        xmax = max(left[0], right[0]) + PADDING_X

        ymin = upper[1] - PADDING_Y
        ymax = lower[1] + PADDING_Y

        xmin = max(0, xmin)
        ymin = max(0, ymin)

        xmax = min(w, xmax)
        ymax = min(h, ymax)

        if xmax > xmin and ymax > ymin:

            eye = frame[ymin:ymax, xmin:xmax]

            if eye.size != 0:

                eye = cv2.resize(eye, (128,128))

                cv2.imshow("Eye Crop", eye)

                # Wait for eyes to settle
                if time.time() - last_move > settle_time:

                    filename = f"{image_counter:06d}.png"

                    cv2.imwrite(
                        os.path.join(SAVE_DIR, filename),
                        eye
                    )

                    with open(LABEL_FILE, "a", newline="") as f:

                        writer = csv.writer(f)

                        writer.writerow([
                            filename,
                            target_x,
                            target_y
                        ])

                    image_counter += 1
                    captured += 1

                    if captured >= captures_per_target:

                        captured = 0

                        target_x = random.randint(
                            margin,
                            screen_w-margin
                        )

                        target_y = random.randint(
                            margin,
                            screen_h-margin
                        )

                        last_move = time.time()

    # -----------------------
    # Draw Target
    # -----------------------

    screen[:] = 0

    cv2.circle(
        screen,
        (target_x,target_y),
        12,
        (0,0,255),
        -1
    )

    cv2.putText(
        screen,
        f"Samples : {image_counter}",
        (30,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.imshow("Target", screen)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()