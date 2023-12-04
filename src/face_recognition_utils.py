import cv2
import os
import numpy as np

def load_known_faces(known_faces_dir):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    x_train = []
    y_labels = []
    current_id = 0
    label_ids = {}

    for root, dirs, files in os.walk(known_faces_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                # Create a label ID for each person
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]

                pil_image = cv2.imread(path)
                gray = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    # Train the recognizer on the faces list and their IDs
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("face-trainner.yml")

    return recognizer, label_ids

def recognize_faces(frame, recognizer, label_ids):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    names = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            name = list(label_ids.keys())[list(label_ids.values()).index(id_)]
        else:
            name = "Unknown"
        names.append(name)

    return faces, names
