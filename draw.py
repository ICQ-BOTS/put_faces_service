import os
import cv2
import numpy as np
import random
import os.path
import dnn_face_detection
import dlib_face_detection
import utils
import traceback

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

BASE_FACE_MODEL_SEGMENT_PATH = os.path.join(
    SCRIPT_PATH, "models", "basic_face_detection", "weight_unet_96.json"
)
BASE_FACE_MODEL_WEIGHTS_PATH = os.path.join(
    SCRIPT_PATH, "models", "basic_face_detection", "weight_unet_96.h5"
)
CLEAR_FACE_MODEL_PATH = os.path.join(
    SCRIPT_PATH, "models", "haar", "haarcascade_frontalface_default.xml"
)
SHAWARMA_MODEL_WEIGHTS_PATH = os.path.join(
    SCRIPT_PATH, "models", "shawarma", "yolov3_training_last.weights"
)

SHAWARMA_MODEL_CONFIG_PATH = os.path.join(
    SCRIPT_PATH, "models", "shawarma", "yolov3_testing.cfg"
)

base_face_model_segment = None
# 1. Модель сегментации человека
def get_base_face_model_segment():
    global base_face_model_segment

    if base_face_model_segment is None:
        from keras.models import model_from_json

        with open(BASE_FACE_MODEL_SEGMENT_PATH, "r") as f:
            base_face_model_segment = model_from_json(f.read())
        base_face_model_segment.load_weights(BASE_FACE_MODEL_WEIGHTS_PATH)

    return base_face_model_segment


# 2. Модель нахождения лица
face_cascade = cv2.CascadeClassifier(CLEAR_FACE_MODEL_PATH)

# 3. Модель нахождения шавермы
shawarma_detection_net = cv2.dnn.readNet(
    SHAWARMA_MODEL_WEIGHTS_PATH, SHAWARMA_MODEL_CONFIG_PATH
)
shawarma_detection_net_layer_names = shawarma_detection_net.getLayerNames()
shawarma_detection_net_output_layers = [
    shawarma_detection_net_layer_names[i[0] - 1]
    for i in shawarma_detection_net.getUnconnectedOutLayers()
]


FACE_BOX_EXPAND_TOP = 0.25
FACE_BOX_EXPAND_BOTTOM = 0.15
FACE_BOX_EXPAND_SIDES = 0.25
SEGMENTATION_CONTEXT_EXPAND = 1.0

PUT_FACTOR = 0.8
FACE_OPACITY = 0.8

ELLIPSE_GRADIENT_PERCENT = 0.1
BOTTOM_GRADIENT = (FACE_BOX_EXPAND_BOTTOM + 0.05) / (
    1.0 + FACE_BOX_EXPAND_TOP + FACE_BOX_EXPAND_BOTTOM
)

HORIZONTAL_ASPECT_RATIO = 3 / 2

img_rows = 96
img_cols = 96
bg_path = r"bg_4.jpg"


def filterMorph(bg_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
    return opening


def erode(bg_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.erode(bg_mask, kernel)
    return opening


def filterConetectedComponents(bg_mask):
    [h, w] = bg_mask.shape[:2]
    max_area = 0
    contours, hierarchy = cv2.findContours(
        bg_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    idx = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            idx = i
    bg_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(bg_mask, contours, idx, (255), -1, 8, hierarchy)
    # cv2.boundingRect(contour)
    return bg_mask


# smooth contour of binary mask border
def smoothContour(bg_mask):
    ksize = int(min(bg_mask.shape[0], bg_mask.shape[1]) / 25)
    if ksize % 2 == 0:
        ksize += 1
    bg_mask = cv2.GaussianBlur(bg_mask, (ksize, ksize), ksize / 3)
    _, bg_mask = cv2.threshold(bg_mask, 127, 255, cv2.THRESH_BINARY)
    return bg_mask


def cropImageToSquare(image):
    [h, w] = image.shape[:2]
    if w > h:
        w0 = int((w - h) / 2)
        image = image[:, w0 : w0 + h, :]
    else:
        h0 = int((h - w) / 4 * 3)
        image = image[h0 : h0 + w, :]
    return image


#                                    top  bottom
def get_face(source_image, v_expand=(0.0, 0.0), h_expand=0.0):
    try:
        base_face_model_segment = get_base_face_model_segment()

        base_shape = source_image.shape[:2]

        # Detect faces
        max_rect = dnn_face_detection.find_face(source_image)
        if max_rect is None:
            max_rect = dlib_face_detection.find_face(source_image)

        if max_rect is None:
            return None, None

        lurb_max_rect = utils.xywh_to_lurb(max_rect)

        source_copy = source_image.copy()

        lurb_max_rect = (0, 0, 100, 100)
        cv2.rectangle(
            source_copy,
            (lurb_max_rect[0], lurb_max_rect[1]),
            (lurb_max_rect[2], lurb_max_rect[3]),
            (255, 0, 0),
            2,
        )

        # expand face box
        base_rect = utils.shape_to_rect(base_shape)
        lurb_base_rect = utils.xywh_to_lurb(base_rect)

        expanded_face_rect = utils.expand_with_borders(
            max_rect,
            base_rect,
            int(max_rect[3] * v_expand[0]),
            int(max_rect[3] * v_expand[1]),
            int(max_rect[2] * h_expand),
            int(max_rect[2] * h_expand),
        )

        lurb_expanded_face_rect = utils.xywh_to_lurb(expanded_face_rect)

        cv2.rectangle(
            source_copy,
            (lurb_expanded_face_rect[0], lurb_expanded_face_rect[1]),
            (lurb_expanded_face_rect[2], lurb_expanded_face_rect[3]),
            (0, 255, 0),
            2,
        )

        face_image = source_image[
            lurb_expanded_face_rect[1] : lurb_expanded_face_rect[3] + 1,
            lurb_expanded_face_rect[0] : lurb_expanded_face_rect[2] + 1,
        ]

        context_face_rect = utils.expand_with_borders(
            expanded_face_rect,
            base_rect,
            int(expanded_face_rect[3] * SEGMENTATION_CONTEXT_EXPAND),
            int(expanded_face_rect[3] * SEGMENTATION_CONTEXT_EXPAND),
            int(expanded_face_rect[2] * SEGMENTATION_CONTEXT_EXPAND),
            int(expanded_face_rect[2] * SEGMENTATION_CONTEXT_EXPAND),
        )

        lurb_context_face_rect = utils.xywh_to_lurb(context_face_rect)

        lurb_context_coords_expanded_face_rect = utils.lurb_conver_to_subrect_coords(
            lurb_expanded_face_rect, lurb_context_face_rect
        )

        face_context_image = source_image[
            lurb_context_face_rect[1] : lurb_context_face_rect[3] + 1,
            lurb_context_face_rect[0] : lurb_context_face_rect[2] + 1,
        ]
        face_context_image_shape = face_context_image.shape[:2]
        # prepare frame for NN
        frame_src = face_context_image
        frame = cv2.resize(frame_src, (img_rows, img_cols))
        frame = frame / 127.5 - 1

        # predict
        inputImage = np.asarray([frame], dtype="float32")
        # Находим пиксеои с человеком
        context_mask = base_face_model_segment.predict(inputImage)[0].squeeze()

        # Умножаем их на 255, чтобы была картинка и убираем лишнее
        context_mask = np.asarray(context_mask * 255, dtype="uint8")
        _, context_mask = cv2.threshold(context_mask, 250, 255, cv2.THRESH_BINARY)

        # Убираем шум морфологическим открытием
        context_mask = filterMorph(context_mask)
        # Находим самый большой контур на изображении и создаём маску по нему
        context_mask = filterConetectedComponents(context_mask)

        context_mask_src = cv2.resize(
            context_mask, (face_context_image_shape[1], face_context_image_shape[0])
        )
        mask_src = context_mask_src[
            lurb_context_coords_expanded_face_rect[
                1
            ] : lurb_context_coords_expanded_face_rect[3]
            + 1,
            lurb_context_coords_expanded_face_rect[
                0
            ] : lurb_context_coords_expanded_face_rect[2]
            + 1,
        ]

        mask_src = smoothContour(mask_src)

        mask_fg = cv2.blur(mask_src, (7, 7))
        mask_fg = cv2.merge([mask_fg, mask_fg, mask_fg]) / 255

        black_image = np.asarray(face_image * mask_fg, dtype="uint8")

        return face_image, mask_fg
    except:
        traceback.print_exc()
        return None, None


def bottom_gradient(mask, size):
    result = mask.copy()
    start = result.shape[0] - size
    for row_index in range(start, result.shape[0]):
        percent = 1.0 - (row_index - start) / size
        result[row_index] = mask[row_index] * (1.0 - (row_index - start) / size)
    return result


def find_shawarma_boxes(shawarma_image):
    global shawarma_detection_net, shawarma_detection_net_output_layers
    base_shape = shawarma_image.shape
    NN_RESIZE = 0.4
    try:
        shawarma_image = cv2.resize(shawarma_image, None, fx=NN_RESIZE, fy=NN_RESIZE)
    except:
        return None
    height, width, channels = shawarma_image.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        shawarma_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )

    shawarma_detection_net.setInput(blob)
    outs = shawarma_detection_net.forward(shawarma_detection_net_output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    result = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(shawarma_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box = boxes[i]
            x = int(max(box[0] / NN_RESIZE, 0))
            y = int(max(box[1] / NN_RESIZE, 0))
            w = int(min(box[2] / NN_RESIZE, base_shape[1] - x - 1))
            h = int(min(box[3] / NN_RESIZE, base_shape[0] - y - 1))

            result.append((x, y, w, h))

    return result


def resize_proportional(image, width=None, height=None, long_side=None):
    if long_side != None:
        if image.shape[0] > image.shape[1]:
            width = None
            height = long_side
        else:
            width = long_side
            height = None
    if height != None:
        width = int(height / image.shape[0] * image.shape[1])
    if width != None:
        height = int(width / image.shape[1] * image.shape[0])

    return cv2.resize(image, (width, height))


def put_image_to_box(dest_image, image, image_mask, box, put_factor=1.0):
    image_aspect_ratio = image.shape[1] / image.shape[0]
    box_aspect_ratio = box[2] / box[3]

    if image_aspect_ratio < box_aspect_ratio:  # put by height
        put_image_width = None
        put_image_height = int(box[3] * put_factor)
    else:  # put by width
        put_image_width = int(box[2] * put_factor)
        put_image_height = None

    resized_image = resize_proportional(image, put_image_width, put_image_height)

    resized_mask = cv2.resize(
        image_mask, (resized_image.shape[1], resized_image.shape[0])
    )

    image_in_box_x = (box[2] - resized_image.shape[1]) // 2
    image_in_box_y = (box[3] - resized_image.shape[0]) // 2
    image_put_x = box[0] + image_in_box_x
    image_put_y = box[1] + image_in_box_y

    inv_mask = 1.0 - resized_mask

    utils.put_image(resized_image, dest_image, image_put_x, image_put_y, resized_mask)

    return


def put_to_shawarma(face_image, shawarma_image):

    mul = 0.5

    # Detect face and segment

    if face_image is None:
        return (1, None)

    face_image, face_mask = get_face(
        cv2.resize(face_image, None, fx=mul, fy=mul),
        (FACE_BOX_EXPAND_TOP, FACE_BOX_EXPAND_BOTTOM),
        FACE_BOX_EXPAND_SIDES,
    )

    if face_image is None:
        return (1, None)

    # Make gradient mask

    bottom_gradient_mask = bottom_gradient(
        face_mask, int(face_image.shape[0] * BOTTOM_GRADIENT)
    )
    # Mix gradient masks
    bottom_gradient_mask = bottom_gradient_mask * FACE_OPACITY

    gradient_dist = ELLIPSE_GRADIENT_PERCENT
    gradient_start_dist = 1.0 - gradient_dist

    x_axis = np.linspace(-1, 1, face_image.shape[1])[None, :]
    y_axis = np.linspace(-1, 1, face_image.shape[0])[:, None]

    radial_gradient_mask = np.minimum(np.sqrt(y_axis ** 2 + x_axis ** 2), 1.0)

    radial_gradient_mask = np.where(
        radial_gradient_mask < gradient_start_dist,
        1.0,
        1.0 - (radial_gradient_mask - gradient_start_dist) / gradient_dist,
    )
    radial_gradient_mask = cv2.merge(
        [radial_gradient_mask, radial_gradient_mask, radial_gradient_mask]
    )
    face_gradient_mask = np.minimum(bottom_gradient_mask, radial_gradient_mask)

    # Find shawarma ellipses

    shawarma_image = resize_proportional(shawarma_image, long_side=800)
    boxes = find_shawarma_boxes(shawarma_image)

    if len(boxes) == 0:
        return (2, None)

    # Put face into ellipses with created mask

    for box in boxes:
        box_aspect_ratio = box[2] / box[3]
        put_face_image = face_image
        if box_aspect_ratio > HORIZONTAL_ASPECT_RATIO:

            if random.random() < 0.5:
                rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
            else:
                rotation = cv2.ROTATE_90_CLOCKWISE
            put_face_image = cv2.rotate(put_face_image, rotation)

        put_image_to_box(
            shawarma_image, put_face_image, face_gradient_mask, box, PUT_FACTOR
        )
    return (0, shawarma_image)


WATERMARK_HEIGHT_PERCENT = 0.1919


def add_watermark(image, watermark_image, watermark_mask):
    watermark_height = int(image.shape[0] * WATERMARK_HEIGHT_PERCENT)
    max_width = int(
        min(
            watermark_height / watermark_image.shape[0] * watermark_image.shape[1],
            image.shape[1],
        )
    )

    resized_watermark = resize_proportional(watermark_image, width=max_width)
    resized_mask = cv2.resize(
        watermark_mask, (resized_watermark.shape[1], resized_watermark.shape[0])
    )
    put_x = (image.shape[1] - resized_watermark.shape[1]) // 2
    put_y = image.shape[0] - resized_watermark.shape[0]

    utils.put_image(resized_watermark, image, put_x, put_y, resized_mask)


def add_bgra_watermark(image, watermark_image):
    add_watermark(result_image, *utils.bgra_to_bgr_and_mask(watermark_image))


if __name__ == "__main__":
    result_image = put_to_shawarma(
        cv2.imread("test.jpg"),
        cv2.imread("shawarma/st_p/green_0.jpg"),
    )
    result_image = result_image[1]
    watermark_image = cv2.imread("watermarks/tests_bot.png", cv2.IMREAD_UNCHANGED)
    add_bgra_watermark(result_image, watermark_image)
    cv2.imshow("Result", result_image)
    cv2.waitKey(-1)
