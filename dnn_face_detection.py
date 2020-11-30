# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import os.path
import utils

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

# defining argument parsers
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())


# defining prototext and caffemodel paths
caffeModel = os.path.join(
    SCRIPT_PATH, "models", "dnn_face", "res10_300x300_ssd_iter_140000.caffemodel"
)
prototextPath = os.path.join(SCRIPT_PATH, "models", "dnn_face", "deploy.prototxt.txt")

# Load Model
print("Loading model...................")
net = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)

BLOB_SIZE = (300, 300)


def find_face(image, required_confidence=0.45):

    # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
    (h, w) = image.shape[:2]
    image_border_box = utils.shape_to_rect(image.shape)

    blob_image = cv2.resize(image, BLOB_SIZE)
    # blobImage convert RGB (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(blob_image, 1.0, BLOB_SIZE, (104.0, 177.0, 123.0))

    # passing blob through the network to detect and pridiction
    net.setInput(blob)
    detections = net.forward()

    max_rect = None
    max_area = None

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence and prediction
        confidence = detections[0, 0, i, 2]

        # filter detections by confidence greater than the minimum     #confidence
        if confidence > required_confidence:
            # compute the (x, y)-coordinates of the bounding box for the
            # object

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            box = box.astype("int")
            box = utils.enable_borders(utils.lurb_to_xywh(box), image_border_box)

            if box is None:
                continue
            (start_x, start_y, width, height) = box

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            width = end_x - start_x
            height = end_y - start_y

            area = width * height

            if (max_rect is None) or (max_area < area):
                max_rect = (start_x, start_y, width, height)
                max_area = area

            # draw the bounding box of the face along Confidance
            # probability

    # show the output image
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)
    return max_rect


# find_face(cv2.imread("test-2.jpg"))
