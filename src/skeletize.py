from msilib.schema import Error
import numpy as np
import os
import cv2
import re
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class KeyStop(Exception): pass
"""
Images from
Barczak, Andre & Reyes, Napoleon & Abastillas, M & Piccio, A & Susnjak, Teo. (2011). A New 2D Static Hand Gesture Colour Image Dataset for ASL Gestures. Res Lett Inf Math Sci. 15. 
The images do not belong to Alten, 
"""

def skeletize(root=r"train_set\\images\\"):
    """
    No surprize, only mediapipe.
    https://google.github.io/mediapipe/solutions/hands#python-solution-api
    """
    # All filenames must follow this match:
    filename_regex = re.compile("hand([0-9])_([0-9]|[a-z])_" + \
                                "(right|left|bot|dif|top)" + \
                                "_seg_[0-9]_cropped\.png"
                               )
    newroot = re.sub("images", "skeletons", root)
    with mp_hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=.5,
            model_complexity=1
            ) as hands:
        fails = 0
        for idx, filename in enumerate(os.listdir(root)):
            newname = re.sub("\.png", ".npy", filename)
            if newname in os.listdir(newroot):
                print("File already treated: SKIPPING")
                continue
            if not filename_regex.match(filename).groups()[1].isalpha():
                print("Not a letter gesture: SKIPPING")
                continue
            image = cv2.flip(cv2.imread(root+filename), 1)
            # Process RGB image
            print("[", end="")
            nb = 0
            failed = False
            while True:
                result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not result.multi_hand_landmarks:
                    print("#", end="")
                    nb += 1
                    if nb == 10:
                        failed = True
                        break
                else:
                    print("]")
                    break
            if failed:
                fails += 1
                print("]")
                print("Fail for:", filename)
                continue
            landmarks = []

            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            cv2.imshow("Check", cv2.flip(annotated_image, 1))
            key = cv2.waitKey(0) & 0xFF
            if key in [27, ord('q')]: raise KeyStop("End of acquisition")
            if key == ord("n"):
                fails += 1
                print("Landmarks refused: SKIPPING")
                continue
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])

            if filename_regex.match(filename):
                print("Saving:", newname)
                np.save(newroot + newname, landmarks)
            else: print("PROBLEM W/ FILENAME, OUT OF REGEX")
    print("Files treated successfuly: ", len(os.listdir(newroot)) - fails)


if __name__ == '__main__':
    skeletize()