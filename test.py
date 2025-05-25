import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

dataset_path = r"C:\Users\Janvi\Desktop\APPROACH1\relativesdataset" 
img_size = (160, 160)
batch_size = 32
epochs = 50

def load_data():
    X = []
    y = []
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    class_to_label = {cls: idx for idx, cls in enumerate(class_names)}

    print("Preprocessing images...")
    for class_name in class_names:
        class_folder = os.path.join(dataset_path, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, img_size)
            img = img.astype('float32') / 255.0

            X.append(img)
            y.append(class_to_label[class_name])

    X = np.array(X)
    y = to_categorical(np.array(y))
    return X, y, class_names

def build_model(input_shape, num_classes):
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))    
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv Block 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Conv Block 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    X, y, class_names = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


    # Save preprocessed data for potential future use
    with open("preprocessed_data.pkl", "wb") as f:
        pickle.dump((X_train, X_val, y_train, y_val, class_names), f)

    input_shape = (img_size[0], img_size[1], 3)
    num_classes = len(class_names)
    model = build_model(input_shape, num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6),
        ModelCheckpoint("cnn_best_model.h5", monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model.save("cnn_final_model.h5")
    print("✅ Model training complete and saved as 'cnn_final_model.h5'")

if __name__ == "__main__":
    main()
