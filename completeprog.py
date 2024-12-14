import cv2
import os
import mediapipe as mp
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

hc = []

# Convert videos to frames
def convert(gesture_folder, target_folder):
    rootPath = os.getcwd()
    majorData = os.path.abspath(target_folder)

    if not os.path.exists(majorData):
        os.makedirs(majorData)

    if os.path.isfile(gesture_folder):
        videos = [gesture_folder]
    else:
        gesture_folder = os.path.abspath(gesture_folder)
        os.chdir(gesture_folder)
        videos = [os.path.join(gesture_folder, video) for video in os.listdir() if os.path.isfile(video)]

    print("Source Directory containing gestures: %s" % (gesture_folder))
    print("Destination Directory containing frames: %s\n" % (majorData))

    for video in tqdm(videos, unit='videos', ascii=True):
        if os.path.isdir(video):
            continue  # Skip directories

        video_name = os.path.abspath(video)
        cap = cv2.VideoCapture(video_name)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lastFrame = None

        gesture_frames_path = os.path.join(majorData, os.path.splitext(os.path.basename(video))[0])
        if not os.path.exists(gesture_frames_path):
            os.makedirs(gesture_frames_path)

        os.chdir(gesture_frames_path)
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            framename = os.path.splitext(os.path.basename(video))[0]
            framename = framename + "frame" + str(count) + ".jpeg"
            hc.append([os.path.join(gesture_frames_path, framename), 'gesture', frameCount])

            if not os.path.exists(framename):
                lastFrame = frame
                cv2.imwrite(framename, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1

        cap.release()
        cv2.destroyAllWindows()

    os.chdir(rootPath)

# Function to resize frames
def resize_frames(input_path, output_path, size=224):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in glob.glob(input_path + '**/*.jpeg', recursive=True):
        img = Image.open(filename).resize((size, size))
        loc = os.path.split(filename)[0]
        subdir = loc.split('/')[-1]  # Adjust for '/' or '\\' as needed
        fullnew_subdir = os.path.join(output_path, subdir)
        if not os.path.exists(fullnew_subdir):
            os.makedirs(fullnew_subdir)
        name = os.path.split(filename)[1]
        img.save(os.path.join(fullnew_subdir, name))

# Augmentations
def brightness(frame, factor_range=(0.85, 1.15)):
    factor = np.random.uniform(factor_range[0], factor_range[1])
    return cv2.convertScaleAbs(frame, alpha=factor, beta=0)

def contrast(frame, factor_range=(0.85, 1.15)):
    factor = np.random.uniform(factor_range[0], factor_range[1])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def noise(frame, noise_level=25, d=9, sigma_color=75, sigma_space=75):
    noise = np.random.normal(0, noise_level, frame.shape).astype(np.uint8)
    noisy_frame = cv2.add(frame, noise)
    return cv2.bilateralFilter(noisy_frame, d, sigma_color, sigma_space)

def augment_frame(frame):
    augmentations = [brightness, contrast, noise]
    num_augmentations = random.randint(2, 4)
    chosen_augmentations = random.sample(augmentations, num_augmentations)

    augmented_frame = frame
    for augmentation in chosen_augmentations:
        augmented_frame = augmentation(augmented_frame)

    return augmented_frame

# MediaPipe Holistic Processing
def process_with_holistic_landmarks(frame, holistic_model, mp_drawing, mp_holistic):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    return frame

# CNN Model for Gesture Recognition
def create_cnn_model(input_shape=(224, 224, 3), num_classes=8):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training the CNN Model
def train_model(train_dir, validation_dir, model):
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=30, zoom_range=0.2)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='sparse')
    val_generator = val_datagen.flow_from_directory(validation_dir, target_size=(224, 224), batch_size=32, class_mode='sparse')

    model.fit(train_generator, epochs=10, validation_data=val_generator)

# Main integration
if __name__ == "__main__":
    gesture_folder = '/path/to/gesture_folder'
    target_folder = '/path/to/target_folder'
    resized_folder = '/path/to/resized_folder'

    # Step 1: Extract frames
    convert(gesture_folder, target_folder)

    # Step 2: Resize extracted frames
    resize_frames(target_folder, resized_folder)

    # Initialize MediaPipe Holistic model
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Process frames with holistic landmarks and apply augmentations
    process_with_holistic_landmarks(gesture_folder, resized_folder, target_folder, size=224)

    # Create CNN model
    model = create_cnn_model(input_shape=(224, 224, 3), num_classes=8)

    # Train the model
    train_model(resized_folder, '/path/to/validation_folder', model)

    # Predict gesture for a manually entered image
    img_path = '/path/to/test_image.jpeg'
    img = Image.open(img_path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    print(f'Predicted class: {predicted_class}')
