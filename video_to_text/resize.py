import cv2
import os
import mediapipe as mp
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import random
import tensorflow as tf

import cv2
import os
import mediapipe as mp
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import random

# Augmentation Functions
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
    num_augmentations = random.randint(2, 4)  # Apply 2 to 4 augmentations
    chosen_augmentations = random.sample(augmentations, num_augmentations)

    augmented_frame = frame
    for augmentation in chosen_augmentations:
        augmented_frame = augmentation(augmented_frame)

    return augmented_frame

# Function to resize and augment frames, then save
def resize_and_augment_frames(input_path, output_path, size=224):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in glob.glob(input_path + '**/*.jpeg', recursive=True):
        img = Image.open(filename)  # Open image
        img_resized = img.resize((size, size))  # Resize the image to desired size
        
        # Convert to NumPy array for augmentation
        img_np = np.array(img_resized)
        
        # Apply augmentations
        augmented_img_np = augment_frame(img_np)
        
        # Convert back to PIL Image after augmentation
        augmented_img = Image.fromarray(augmented_img_np)

        # Create subdirectory if needed
        loc = os.path.split(filename)[0]
        subdir = loc.split('/')[-1]  # Get subdirectory name
        fullnew_subdir = os.path.join(output_path, subdir)
        if not os.path.exists(fullnew_subdir):
            os.makedirs(fullnew_subdir)

        # Save the augmented and resized frame
        name = os.path.split(filename)[1]
        augmented_img.save(os.path.join(fullnew_subdir, name))


# Holistic landmark processing function
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
