import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle

dataset_path = r"C:\Users\Janvi\Desktop\APPROACH1\relativesdataset"  # Folder with PatientX_RelativeY folders  jese ki patient1_relative1
img_size = (160, 160)

X = []
y = []
class_names = sorted(os.listdir(dataset_path))  # Ensure consistent label order ....sorted data taki better rhe chronology

# Map class names to numeric labels
class_to_label = {cls: idx for idx, cls in enumerate(class_names)}   #patient1_relative1 : 0  -> aise ho rha h

print("Preprocessing images...")

for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_folder):
        continue

    for filename in os.listdir(class_folder):
        img_path = os.path.join(class_folder, filename)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Resize and normalize
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0

        X.append(img)
        y.append(class_to_label[class_name])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# One-hot encode labels
y = to_categorical(y)  

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


with open("preprocessed_data.pkl", "wb") as f:
    pickle.dump((X_train, X_val, y_train, y_val, class_names), f)

print("Saved preprocessed data to 'preprocessed_data.pkl'")
