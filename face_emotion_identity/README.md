Emotion & Identity Detection using Computer Vision

Project Overview

This project is a real-time 'Emotion and Face Identity Detection System' that leverages deep learning and computer vision techniques to detect human facial expressions and recognize identities from a webcam feed.

It combines:
- Facial Expression Recognition - using a custom-trained CNN model.
- Face Recognition - using the `DeepFace` library.
- Live Webcam Feed Processing - using OpenCV.

Group Members

1. Kaab Gazdar  
2. Ishaan Joshi  
3. Keerthi Venkata  

Requirements

Make sure the following libraries are installed in your Python environment:

  pip install opencv-python keras tensorflow tensorflow-datasets deepface numpy pillow

Additional dependencies:
- `haarcascade_frontalface_default.xml` is used (automatically accessed via OpenCV).
- A trained emotion model: `emotion_model.h5` (You must place it in the correct path as mentioned in the script).

Directory Structure

face_emotion_identity/
├── known_faces/              # Folder created dynamically to store images from LFW dataset
│   └── <person_name>/        # Each person's folder with up to 2 images
│       └── 1.jpg
├── emotion_model.h5          # Pre-trained emotion recognition model
├── new.py                    # Main execution script
└── README.md                 # This file

How to Run

1. Clone the project or copy the code into a Python script named `new.py`.
2. Place your `emotion_model.h5` in the appropriate directory and update its path in the code.
3. Run the script:
    python script.py
4. A webcam window will open. Press 'Q' to quit.

Code Explanation

1. Build Known Faces

  build_known_faces(num_people=5, images_per_person=2)

- Downloads the 'Labeled Faces in the Wild (LFW)' dataset.
- Stores `num_people` unique identities with up to `images_per_person` images in the `known_faces/` folder.

2. Build Face Database

  build_face_db(folder="known_faces")

- Loads one image per person to create a dictionary that maps names to image paths for face verification.

3. Emotion Detection

model = load_model("emotion_model.h5")

- A CNN model is used to classify facial expressions into:
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

4. Face Detection

face_classifier = cv2.CascadeClassifier(...)

- Uses Haar Cascades to detect faces in the webcam feed.

5. Face Verification

DeepFace.verify(img1_path=temp_img, img2_path=db_img_path)

- Compares detected face with known images using deep learning-based face verification.

6. Real-Time Webcam Processing

- Frames are read from the webcam using OpenCV.
- For each detected face:
  - Expression is classified.
  - Identity is verified.
  - Both results are annotated live on the screen.

Theory Topics

- Face Detection: Haar Cascade Classifier (Viola-Jones Algorithm).
- Emotion Recognition: Convolutional Neural Networks (CNNs) trained on grayscale facial images.
- Face Verification: DeepFace leverages models like VGG-Face, Facenet, etc., for embedding-based verification.
- Dataset Used: LFW (Labeled Faces in the Wild).
- Libraries:
  - `OpenCV`: Image capturing and processing.
  - `Keras/TensorFlow`: Emotion model loading and inference.
  - `DeepFace`: Identity verification.
  - `TensorFlow Datasets`: Loading the LFW dataset.

Notes

- Ensure good lighting conditions for accurate detection.
- Identity recognition is sensitive to face alignment and image quality.
- The `DeepFace.verify` function is computationally expensive—this can cause slight delays.

Future Work

- Improve real-time performance using face embeddings caching.
- Use MTCNN or RetinaFace for better face detection.
- Replace Haar cascades with DNN-based detectors for more robust detection.
- Train emotion model with a larger, more diverse dataset like FER2013.

License

This project is for educational purposes only.