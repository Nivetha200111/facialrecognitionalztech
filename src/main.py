import cv2
import os
import face_recognition_utils as fru

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    known_faces_dir = os.path.join(base_dir, 'data', 'known_faces')
    recognizer, label_ids = fru.load_known_faces(known_faces_dir)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        # Convert the image from BGR color (which OpenCV uses) to gray color
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use OpenCV recognizer to recognize faces
        faces, names = fru.recognize_faces(frame, recognizer, label_ids)

        for ((x, y, w, h), name) in zip(faces, names):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
