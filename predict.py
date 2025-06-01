import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logging
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

model_path = 'cnn_best_model.h5'  
data_pickle = 'preprocessed_data.pkl'  # to load class_names
img_size = (160, 160)

model = load_model(model_path)

with open(data_pickle, 'rb') as f:
    _, _, _, _, class_names = pickle.load(f)

# Load kia h haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, img_size)
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)  
    return face_img

def predict_face(face_img):
    processed = preprocess_face(face_img)
    preds = model.predict(processed, verbose=0)  # Added verbose=0 to suppress prediction output
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    return class_names[class_idx], confidence

def main():
    cap = cv2.VideoCapture(0)  # Camera khulega

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting webcam. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            label, conf = predict_face(face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{label} ({conf*100:.1f}%)"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
