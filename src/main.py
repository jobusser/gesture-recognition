import argparse
import csv
import copy
import itertools
from collections import Counter
from collections import deque

import cv2
import mediapipe as mp

from ui import Drawer

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=int, default=0.5)
    parser.add_argument("--show_image", action='store_true')

    args = parser.parse_args()

    return args




def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    show_image = args.show_image


    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)


    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
    )

    drawer = Drawer(mp_hands)

    while True:
        # camera capture
        success, image = cap.read()
        if not success:
            print("No camera found.")
            break
        image = cv2.flip(image, 1) # mirror display

        # gesture recognition
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)


        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                print('HAND LANDMARKS: ', hand_landmarks)
                print('HANDEDNESS: ', handedness)

        if show_image:
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                image = drawer.draw_landmarks(image, results)

            key = cv2.waitKey(10)
            if key == 27: # ESC
                break
            image = drawer.draw_ui(image)
            cv2.imshow('Gesture recognition', image)

    cap.release()
    if show_image:
        cv2.destroyAllWindows()


# TODO:
# landmark list, draw landmarks to make it look unique
# text above to show classification
# start logging data with different modes
# auto train model

if __name__ == '__main__':
    main()

















