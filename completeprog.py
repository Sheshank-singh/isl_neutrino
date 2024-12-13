import cv2
import os
import mediapipe as mp
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from os.path import join, exists
import random

hc = []


def convert(gesture_folder, target_folder):
    rootPath = os.getcwd()
    majorData = os.path.abspath(target_folder)

    if not exists(majorData):
        os.makedirs(majorData)

    # If gesture_folder is a file, process it directly
    if os.path.isfile(gesture_folder):
        videos = [gesture_folder]
    else:
        gesture_folder = os.path.abspath(gesture_folder)
        os.chdir(gesture_folder)
        videos = [join(gesture_folder, video) for video in os.listdir() if os.path.isfile(video)]

    print("Source Directory containing gestures: %s" % (gesture_folder))
    print("Destination Directory containing frames: %s\n" % (majorData))

    for video in tqdm(videos, unit='videos', ascii=True):
        if os.path.isdir(video):
            continue  # Skip directories (just in case)
        
        # Process video file
        video_name = os.path.abspath(video)
        cap = cv2.VideoCapture(video_name)  # capturing input video
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lastFrame = None

        gesture_frames_path = os.path.join(majorData, os.path.splitext(os.path.basename(video))[0])
        if not os.path.exists(gesture_frames_path):
            os.makedirs(gesture_frames_path)

        os.chdir(gesture_frames_path)
        count = 0

        while True:
            ret, frame = cap.read()  # extract frame
            if ret is False:
                break
            framename = os.path.splitext(os.path.basename(video))[0]
            framename = framename + "frame" + str(count) + ".jpeg"
            hc.append([join(gesture_frames_path, framename), 'gesture', frameCount])

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
        
def brightness(frame, factor_range=(0.85, 1.15)):
    factor = np.random.uniform(factor_range[0], factor_range[1])
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)
    return adjusted_frame

def contrast(frame, factor_range=(0.85, 1.15)):
    factor = np.random.uniform(factor_range[0], factor_range[1])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = np.clip(hsv[..., 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def noise(frame, noise_level=25, d=9, sigma_color=75, sigma_space=75):
    noise = np.random.normal(0, noise_level, frame.shape).astype(np.uint8)
    noisy_frame = cv2.add(frame, noise)
    filtered_frame = cv2.bilateralFilter(noisy_frame, d, sigma_color, sigma_space)
    return np.clip(filtered_frame, 0, 255).astype(np.uint8)

def augment_frame(frame):
    augmentations = [brightness, contrast, noise]
    num_augmentations = random.randint(2, 4)  # Randomly choose 2-4 augmentations
    chosen_augmentations = random.sample(augmentations, num_augmentations)

    augmented_frame = frame
    for augmentation in chosen_augmentations:
        augmented_frame = augmentation(augmented_frame)  # Apply each selected augmentation

    return augmented_frame


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

def process_videos(input_video_folder, resize_folder, target_folder, size=224):
    # Initialize MediaPipe Holistic model
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    if not os.path.exists(resize_folder):
        os.makedirs(resize_folder)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Process each video in the gesture folder
    for video_path in glob.glob(input_video_folder + '**/*.mp4', recursive=True):  # or other formats
        video_name = os.path.basename(video_path).split('.')[0]
        
        # Extract frames from the video
        video_resize_folder = os.path.join(resize_folder, video_name)
        convert(video_path, video_resize_folder, size)

        # Now process each extracted frame
        for frame_filename in glob.glob(video_resize_folder + '/*.jpeg'):
            frame = cv2.imread(frame_filename)

            # Apply augmentations
            augmented_frame = augment_frame(frame)

            # Process with holistic landmarks
            augmented_frame_with_landmarks = process_with_holistic_landmarks(augmented_frame, holistic_model)

            # Save the augmented frame with landmarks
            target_frame_filename = os.path.join(target_folder, os.path.basename(frame_filename))
            cv2.imwrite(target_frame_filename, augmented_frame_with_landmarks)

# Main integration
if __name__ == "__main__":
    gesture_folder = '/Users/shriya/Documents/GitHub/isl_neutrino/gesture_folder/gesture1'
    target_folder = '/Users/shriya/Documents/GitHub/isl_neutrino/target_folder/newvideo'
    resized_folder = '/Users/shriya/Documents/GitHub/isl_neutrino/resized_folder'

    # Step 1: Extract frames
    convert(gesture_folder, target_folder)

    # Step 2: Resize extracted frames
    resize_frames(target_folder, resized_folder)

    process_videos(gesture_folder, resized_folder, target_folder, size=224)
