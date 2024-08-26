import cv2
import mediapipe as mp
from ui.fps_calc import FpsCalc

class Drawer(object):
    def __init__(self, mp_hands, google_draw=False):
        self.mp_hands = mp_hands

        self.fps_calculator = FpsCalc(buffer_len=10)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # config
        self.google_draw = google_draw

    def draw_ui(self, image):
        image = self.draw_fps(image)
        return image


    def draw_fps(self, image):
        fps = self.fps_calculator.get()
        fps_text = f"FPS: {int(fps)}"
        height, width, _ = image.shape

        # bottom right
        text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = width - text_size[0] - 10
        text_y = height - 10

        cv2.putText(image, fps_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def draw_landmarks(self, image, results):
        if results.multi_hand_landmarks is None:
            return image

        if self.google_draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

        else:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_image_coords = calc_landmark_image_coords(image, hand_landmarks)
                image = draw_hand(image, landmark_image_coords)


        return image




def calc_landmark_image_coords(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def draw_hand(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 1)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 1)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 1)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 1)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 1)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 1)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 1)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 1: 
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 2:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 3:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 4: 
            cv2.circle(image, (landmark[0], landmark[1]), 8, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), 1)
        if index == 5:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 6:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 7:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 8: 
            cv2.circle(image, (landmark[0], landmark[1]), 8, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), 1)
        if index == 9: 
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 10:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 11:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 12:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), 1)
        if index == 13:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 14:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 15:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 16:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), 1)
        if index == 17:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 18:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 19:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), 1)
        if index == 20:
            cv2.circle(image, (landmark[0], landmark[1]), 8, (241, 255, 51),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), 1)

    return image

