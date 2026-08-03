import cv2
import mediapipe as mp
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)
RIGHT_EYE = [33, 133, 159, 145, 153, 154, 155]
LEFT_EYE = [362, 263, 386, 374, 380, 381, 382]

RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]


while True:
    success, frame = cap.read()
    if not success:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        ''' #proto type 1
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION
            )'''
        face = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
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
        for iris in [RIGHT_IRIS, LEFT_IRIS]:
            iris_points=[]
            for idx in iris:
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
            center_x = int(sum([p[0] for p in iris_points]) // len(iris_points))
            center_y = int(sum([p[1] for p in iris_points]) // len(iris_points))
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


        
    cv2.imshow("Eye and Iris Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()