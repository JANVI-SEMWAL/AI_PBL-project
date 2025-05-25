import cv2
import os

# === CONFIGURATION ===
dataset_path = r"C:\Users\Janvi\Desktop\APPROACH1\relativesdataset"  # Existing dataset folder
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Loop through each relative folder
for relative_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, relative_folder)

    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)

            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {image_path}")
                continue

            # Convert to grayscale for detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # If face found, crop and overwrite
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = img[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (160, 160))
                    cv2.imwrite(image_path, face_resized)
                    break  # Only save the first face
            else:
                print(f"❌ No face found in: {image_path} (image not modified)")

        print(f"✅ Processed folder: {relative_folder}")

print("\n🎉 All faces cleaned and saved in-place in the 'relatives' dataset.")
