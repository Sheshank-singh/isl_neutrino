"""import cv2
import os
import mediapipe as mp
import pickle
from os.path import join, exists
import hs
import argparse
from tqdm import tqdm

hc = []


def convert(gesture_folder, target_folder):
    rootPath = os.getcwd()
    majorData = os.path.abspath(target_folder)

    if not exists(majorData):
        os.makedirs(majorData)

    gesture_folder = os.path.abspath(gesture_folder)

    os.chdir(gesture_folder)
    gestures = os.listdir(os.getcwd())

    print("Source Directory containing gestures: %s" % (gesture_folder))
    print("Destination Directory containing frames: %s\n" % (majorData))

    for gesture in tqdm(gestures, unit='actions', ascii=True):
        gesture_path = os.path.join(gesture_folder, gesture)
        os.chdir(gesture_path)

        gesture_frames_path = os.path.join(majorData, gesture)
        if not os.path.exists(gesture_frames_path):
            os.makedirs(gesture_frames_path)

        videos = os.listdir(os.getcwd())
        videos = [video for video in videos if(os.path.isfile(video))]

        for video in tqdm(videos, unit='videos', ascii=True):
            name = os.path.abspath(video)
            cap = cv2.VideoCapture(name)  # capturing input video
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            lastFrame = None

            os.chdir(gesture_frames_path)
            count = 0

            while True:
                ret, frame = cap.read()  # extract frame
                if ret is False:
                    break
                framename = os.path.splitext(video)[0]
                framename = framename + "_frame_" + str(count) + ".jpeg"
                hc.append([join(gesture_frames_path, framename), gesture, frameCount])

                if not os.path.exists(framename):
                    frame = hs.handsegment(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    lastFrame = frame
                    cv2.imwrite(framename, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                count += 1

            os.chdir(gesture_path)
            cap.release()
            cv2.destroyAllWindows()

    os.chdir(rootPath)


if __name__ == '__main__':
    gesture_folder = '/Users/shriya/Documents/GitHub/isl_neutrino/gesture_folder/gesture1/newvideo.mov'
    target_folder = '/Users/shriya/Documents/GitHub/isl_neutrino/target_folder'

    if not os.path.exists(gesture_folder):
        print(f"Error: Gesture folder '{gesture_folder}' does not exist.")
        exit(1)

    if not os.path.exists(target_folder):
        print(f"Creating target folder '{target_folder}'...")
        os.makedirs(target_folder)

    convert(gesture_folder, target_folder)
  """
   
import cv2
import os
import mediapipe as mp
import pickle
from os.path import join, exists
import hs
import argparse
from tqdm import tqdm

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
    