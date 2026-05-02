import cv2
import numpy as np
from keras.models import load_model
from deepface import DeepFace
import os
import tensorflow_datasets as tfds
from PIL import Image

def build_known_faces(num_people=5, images_per_person=2, output_folder="known_faces"):
    os.makedirs(output_folder, exist_ok=True)
    ds = tfds.load("lfw", split="train", shuffle_files=False)
    added = {}
    for example in tfds.as_numpy(ds):
        label = example["label"].decode("utf-8")
        img = example["image"]
        if added.get(label, 0) >= images_per_person:
            continue
        person_dir = os.path.join(output_folder, label)
        os.makedirs(person_dir, exist_ok=True)
        idx = added.get(label, 0) + 1
        Image.fromarray(img).save(os.path.join(person_dir, f"{idx}.jpg"))
        added[label] = idx
        if len(added) >= num_people:
            break
    print(f"Created {len(added)} folders in '{output_folder}' with up to {images_per_person} images each.")

def build_face_db(folder="known_faces"):
    db = {}
    for person_name in os.listdir(folder):
        person_dir = os.path.join(folder, person_name)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                db[person_name] = img_path
                break
    return db

if __name__ == "__main__":
    build_known_faces()

model = load_model(r"X:\DIT\Sem1\CV\face_emotion_identity\face_emotion_identity\emotion_model.h5", compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_db = build_face_db()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    recognized_name = "Unknown"
    identity_checked = False

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)

        if np.sum([roi_resized]) != 0:
            roi_normalized = roi_resized.astype('float') / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=0)
            roi_input = np.expand_dims(roi_input, axis=-1)

            prediction = model.predict(roi_input)[0]
            emotion = emotion_labels[prediction.argmax()]

            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if not identity_checked:
                identity_checked = True
                temp_img = "temp_face.jpg"
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(temp_img, rgb_frame)

                for name, db_img_path in face_db.items():
                    try:
                        result = DeepFace.verify(img1_path=temp_img, img2_path=db_img_path, enforce_detection=False)
                        if result["verified"]:
                            recognized_name = name
                            break
                    except:
                        continue

                if os.path.exists(temp_img):
                    os.remove(temp_img)

        cv2.putText(frame, recognized_name, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    cv2.imshow("Emotion & Identity Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()