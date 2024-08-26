import cv2
import mediapipe as mp
from ui.fps_calc import FpsCalc

class Drawer(object):
    def __init__(self, mp_hands):
        self.mp_hands = mp_hands

        self.fps_calculator = FpsCalc(buffer_len=10)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def draw_ui(self, image, results):
        image = self.draw_fps(image)
        image = self.draw_landmarks(image, results)
        return image


    def draw_fps(self, image):
        fps = self.fps_calculator.get()
        fps_text = f"FPS: {int(fps)}"
        height, width, _ = image.shape

        # bottom right
        text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = width - text_size[0] - 10
        text_y = height - 10

        cv2.putText(image, fps_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return image

    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        return image

