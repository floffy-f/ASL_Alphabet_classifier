import numpy as np
import sklearn.neighbors as knn
import os
import re
import cv2
import mediapipe as mp
import webbrowser as wb
from deprecated.sphinx import deprecated
import pygame
from moviepy.editor import VideoFileClip

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Error class to interupt capture
class EndCapture(Exception): pass

THUMB_PAIRS  = [ (0, 1)
               , (1, 2)
               , (2, 3)
               , (3, 4)
               ]
INDEX_PAIRS  = [ (0, 5)
               , (5, 6)
               , (6, 7)
               , (7, 8)
               ]
MIDDLE_PAIRS = [ (0, 9)
               , (9, 10)
               , (10, 11)
               , (11, 12)
               ]
RING_PAIRS   = [ (0, 13)
               , (13, 14)
               , (14, 15)
               , (15, 16)
               ]
PINKY_PAIRS  = [ (0, 17)
               , (17, 18)
               , (18, 19)
               , (19, 20)
               ]
JOINTS_PAIRS = np.array(THUMB_PAIRS + INDEX_PAIRS + MIDDLE_PAIRS + RING_PAIRS + PINKY_PAIRS)

@deprecated(version="1.0", reason="Angles computation has been chosen")
def flip_y(hand):
    """
    Flip y axis for left-right hand change
    """
    return np.stack((1. - hand[:, 0], hand[:, 1]), axis=1)

THUMB_TRIPLET =  [ (1, 0, 2)
                 , (2, 1, 3)
                 , (3, 2, 4)
                 ]
INDEX_TRIPLET =  [ (5, 0, 6)
                 , (6, 5, 7)
                 , (7, 6, 8)
                 ]
MIDDLE_TRIPLET = [ (9, 0, 10)
                 , (10, 9, 11)
                 , (11, 10, 12)
                 ]
RING_TRIPLET  =  [ (13, 0, 14)
                 , (14, 13, 15)
                 , (15, 14, 16)
                 ]
PINKY_TRIPLET =  [ (17, 0, 18)
                 , (18, 17, 19)
                 , (19, 18, 20)
                 ]
PALM_TRIPLET  =  [ (0, 1, 5)
                 , (0, 5, 9)
                 , (0, 9, 13)
                 , (0, 13, 17)
                 ]

HAND_TRIPLETS = THUMB_TRIPLET  \
              + INDEX_TRIPLET  \
              + MIDDLE_TRIPLET \
              + RING_TRIPLET   \
              + PINKY_TRIPLET  \
              + PALM_TRIPLET
def compute_angle(hand, i, j, k):
    r"""
    Computes angles between segments ij and ik in data_sequence.
    Formula:
    .. math::
        :label: angles
            cos(ij, ik) = (ij \dot ik)/(||ij||.||ik||)
    
    :param i: angle point
    :param j: first extremity
    :param k: second extremity
    
    :returns: angle <ij,jk>
    """
    vecs = hand[(j, k), :] - hand[(i, i), :]
    
    # Dot prod and norm
    inner = vecs[0, :] @ vecs[1, :]
    norm = np.prod(np.linalg.norm(vecs, axis=1))

    # If i and j or i and k are in superposition, norms is going to contain
    #     zeros, so we can't make the division.
    if norm < 1e-8:
        inner = 0.
        norm = 1.

    return np.arccos(inner / norm)

def angles(hand):
    """
    Gets angles between joints and fingers (including in palm)
    """
    return np.array([compute_angle(hand, i, j, k) for i, j, k in HAND_TRIPLETS]) / np.pi

@deprecated(version="1.0", reason="Angles computation has been chosen")
def displacement_vecs(hand):
    """
    Computes the vectors between joints (not image-position dependant anymore)

    :returns: list of pairwise joints distances, and angles in the image frame of
        the joints segments (to the width axis). The distances are squared.
    """
    vectors = hand[JOINTS_PAIRS[:, 0]] - hand[JOINTS_PAIRS[:, 1]]

    # Polar representation:
    # [rho_1^2, ..., rho_21^2, theta_1, ..., theta_21]
    polar = np.stack(( (vectors * vectors).sum(axis=1)
                     , np.arctan(vectors[:, 1] / vectors[:, 0])
                     ),
                     axis=0)
    return polar.flatten()

@deprecated(version="1.0", reason="Angles computation has been chosen")
def just_flatten(hand):
    maxx = np.max(hand[0, :])
    minx = np.min(hand[0, :])
    maxy = np.max(hand[1, :])
    miny = np.min(hand[1, :])
    pre_treat = (hand - np.array([[minx, miny]]))  / np.array([[maxx - minx, maxy - miny]])
    return pre_treat.flatten()

@deprecated(version="1.0", reason="Angles computation has been chosen")
def select_fingers(hand):
    return just_flatten(hand[1:])

def pre_treatment(hand):
    return angles(hand)

def char_to_class(char: str):
    """
    Class in [0, 10+26=36)
    10 first for digits, 26 others for letters
    """
    o = ord(char)
    if 47 < o < 97:
        # Digit
        raise ValueError("No digits")
        #return int(char)
    elif 96 < o < 123:
        # Letter
        return o - 97
    else:
        raise ValueError("Matching pbm w/ the regex")


def class_to_char(class_num: int):
    """
    Inverse of char_to_class
    """
    # if 0 <= class_num < 10:
    #     return str(class_num)
    # elif class_num > 9:
    #     return chr(class_num + 87)
    if 0 <= class_num < 26:
        return chr(class_num + 97)
    else:
        raise ValueError("Pbm with the knn cardinality w/ class number: " + str(class_num))
        
DISCRIMINATE_FACTOR = .8
def discriminate_a(hand, letter):
    """
    Consider that knn is wrong if the thumb angle is narrow
    """
    if letter == "a":
        alpha = compute_angle(hand, 3, 2, 4)
        if alpha < DISCRIMINATE_FACTOR * np.pi: return "s"
        else: return "a"
    else: return letter

class HandModelKNN:
    LIGHTS = {
        "right": 0,
        "left": 1,
        "bot": 2,
        "top": 3,
        "dif": 4,
    }
    def __init__(self, k, URL, loading=False, root = "train_set\\skeletons\\", word= "saw"):
        self.k = k
        self.complete_word = word

        self.gauge = 0
        self.URL = URL
        self.stop = len(self.complete_word)

        self.vecs = []
        self.labels = []
        self.participant = []
        self.light = []

        title_regex = re.compile("hand([0-9])_([0-9]|[a-z])_" + \
                                 "(right|left|bot|dif|top)" + \
                                 "_seg_[0-9]_cropped\.npy",
                                 re.I
                                )

        if loading:
            for filename in os.listdir(root):
                x, g, ill = title_regex.match(filename).groups()
                if 96 < ord(g) < 123:
                    # If letter on gesture, treat it
                    hand_array = np.load(root + filename)

                    l_features_vec = pre_treatment(hand_array)
                    # Left hand
                    self.vecs.append(l_features_vec)
                    self.labels.append(char_to_class(g))
                    self.participant.append(x)
                    self.light.append(HandModelKNN.LIGHTS[ill])

            print("Save full matrix")
            np.save("train_set/train_matrix.npy", self.vecs)
            np.save("train_set/train_labels.npy", self.labels)
        else:
            self.vecs   = np.load("train_set/train_matrix.npy")
            self.labels = np.load("train_set/train_labels.npy")
        self.build_knn()

    
    def build_knn(self):
        if self.vecs is None:
            raise ValueError("Error while computing vetors: no skeletons in files")
        else:
            acc = []
            models = []
            # for c in np.linspace(0.1, 2., 10):
            for k in range(1, 10):
                candidate = knn.KNeighborsClassifier(
                    n_neighbors=k,
                    weights='distance',
                    metric='l2' # L1 distance (rho is already squared)
                )
                print(f"Fitting model for k={k}...", end=" ")
                candidate.fit(self.vecs, self.labels)
                print("Done!")
                sc = candidate.score(self.vecs, self.labels)
                print("Accuracy:", sc)
                acc.append(sc)
                models.append(candidate)
            self.knn = models[np.argmax(acc)]

    
    def read_cam(self):
        print("###### Reading camera ######")
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=.4,
                    model_complexity=1,
                    min_tracking_confidence=.7
                ) as hands:
            while cap.isOpened():
                success, img = cap.read()
                if not success: continue
                img.flags.writeable = False
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img)

                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        hand_array = np.array([
                            [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
                            ])
                        class_pred = self.knn.predict([pre_treatment(hand_array)])[0]
                        proba = np.max(
                            self.knn.predict_proba([pre_treatment(hand_array)])
                        )
                        img = cv2.flip(img, 1)
                        letter = discriminate_a(hand_array, class_to_char(class_pred))
                        cv2.putText( img
                                   , letter + "  (~{:.1f}%)".format(proba * 100.)
                                   , (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                                   )
                        self.wait_word(letter)
                        if self.gauge == self.stop:
                            self.open_browser()
                            break
                else:
                    img = cv2.flip(img, 1)
                cv2.imshow('Show hands', img)
                if cv2.waitKey(5) & 0xFF in (27, 113):
                    # <esc> or Q => quit
                    break
                
        print("###### Reading finish ######")

    def wait_word(self, letter):
        if letter == self.complete_word[self.gauge]:
            self.gauge += 1

    def open_browser(self, default='chrome'):
        try:
            wb.get(default).open(self.URL, new=2)
        except:
            print("Browser", default, "is not available, swithing to first available instance.")
            wb.open(self.URL)
    
    def open_video(self):
        """
        https://codeloop.org/how-to-play-mp4-videos-in-python-pyglet/
        """
        # Destroy previous windows
        cv2.destroyAllWindows()
        VideoFileClip('saw-video.mp4').preview()

if __name__ == '__main__':
    instance = HandModelKNN(1, "", loading=True)
    instance.read_cam()