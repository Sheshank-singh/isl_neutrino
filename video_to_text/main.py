from video_extract import convert
from resize import resize_and_augment_frames, process_with_holistic_landmarks
from model import create_cnn_model
import mediapipe as mp

if __name__ == "__main__":
    gesture_folder = '/Users/shriya/Documents/GitHub/isl_neutrino/gesture_folder/gesture1'
    target_folder = '/Users/shriya/Documents/GitHub/isl_neutrino/target_folder'
    resized_folder = '/Users/shriya/Documents/GitHub/isl_neutrino/resized_folder'

    # Step 1: Extract frames
    convert(gesture_folder, target_folder)

    # Step 2: Resize extracted frames
    resize_and_augment_frames(target_folder, resized_folder)

    # Initialize MediaPipe Holistic model
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Process frames with holistic landmarks and apply augmentations
    process_with_holistic_landmarks(gesture_folder, resized_folder, target_folder, size=224)

    # Create CNN model
    model = create_cnn_model(input_shape=(224, 224, 3), num_classes=8)

    # Train the model
    """train_model(resized_folder, '/path/to/validation_folder', model)"""

    # Predict gesture for a manually entered image
    """img_path = '/path/to/test_image.jpeg'
    img = Image.open(img_path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    print(f'Predicted class: {predicted_class}')"""
