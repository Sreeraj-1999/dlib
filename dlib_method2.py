import cv2
import dlib
import numpy as np
import os

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('D:\DL_projrcts\Video_Person\shape_predictor_68_face_landmarks.dat')

face_recognizer = dlib.face_recognition_model_v1("D:\DL_projrcts\Video_Person\dlib_face_recognition_resnet_model_v1.dat")

known_face_descriptors = {}
output_folder = 'detected_faces'
os.makedirs(output_folder, exist_ok=True)

video_capture = cv2.VideoCapture('D:\DL_projrcts\Video_Person\WhatsApp Video 2023-09-20 at 17.49.23.mp4')

face_descriptor = None

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray_frame)

    for face in faces:
        shape = predictor(gray_frame, face)

        face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)

        matched = False
        for name, descriptor in known_face_descriptors.items():

            distance = np.linalg.norm(np.array(face_descriptor) - np.array(descriptor))
            if distance < 0.6:  
                matched = True
                break

        if not matched:

            x, y, w, h = face.left(), face.top(), face.width(), face.height()

            face_image = frame[y:y + h, x:x + w]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            name = f'Face_{len(known_face_descriptors) + 1}'
            known_face_descriptors[name] = face_descriptor

            cv2.imwrite(os.path.join(output_folder, f'{name}.jpg'), face_image)


    face_count = len(known_face_descriptors)
    cv2.putText(frame, f'Faces Detected: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

print(f'Total Faces Counted: {len(known_face_descriptors)}')
