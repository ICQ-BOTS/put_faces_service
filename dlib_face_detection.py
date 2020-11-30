# Import the necessary libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib
import utils
from imutils import face_utils
import imutils
import os
import os.path

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

PREDICTOR_PATH = os.path.join(
    SCRIPT_PATH,
    "models",
    "dlib_face_landmarks",
    "shape_predictor_68_face_landmarks.dat",
)


def find_face(image):
    image_borders_rect = utils.shape_to_rect(image.shape)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    normal_rects = []

    max_rect = None
    max_area = None

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        rect = utils.enable_borders(
            (rect.left(), rect.top(), rect.width(), rect.height()), image_borders_rect
        )
        if rect is None:
            continue

        area = rect[2] * rect[3]

        if max_rect is None:
            max_rect = rect
        elif max_area < area:
            max_rect = rect

    if max_rect is None:
        return None

    source_copy = image.copy()
    return max_rect


if __name__ == "__main__":
    #  Loading the image to be tested
    test_image = cv2.imread("test-2.jpg")
    find_face(test_image)
