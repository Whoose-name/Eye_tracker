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
while True:
    success, frame = cap.read()
    if not success:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_face.FACEMESH_TESSELATION
            )
    cv2.imshow("Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()