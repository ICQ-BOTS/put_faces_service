import numpy as np
import cv2


def float_rect_to_int(rect):
    return (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))


def xywh_to_lurb(rect):
    return (rect[0], rect[1], rect[0] + rect[2] - 1, rect[1] + rect[3] - 1)


def lurb_to_xywh(rect):
    return (rect[0], rect[1], rect[2] - rect[0] + 1, rect[3] - rect[1] + 1)


def expand(rect, up=0, bottom=0, left=0, right=0):
    result = xywh_to_lurb(rect)

    return lurb_to_xywh(
        (result[0] - left, result[1] - up, result[2] + right, result[3] + bottom)
    )


def shape_to_rect(shape):
    #              cols       rows
    return (0, 0, shape[1], shape[0])


def enable_borders(rect, borders):
    lurb_rect = xywh_to_lurb(rect)
    lurb_borders = xywh_to_lurb(borders)
    if (
        (lurb_rect[0] > lurb_borders[2])
        or (lurb_rect[1] > lurb_borders[3])
        or (lurb_rect[2] < lurb_borders[0])
        or (lurb_rect[3] < lurb_borders[1])
    ):
        return None

    return lurb_to_xywh(
        (
            max(lurb_rect[0], lurb_borders[0]),
            max(lurb_rect[1], lurb_borders[1]),
            min(lurb_rect[2], lurb_borders[2]),
            min(lurb_rect[3], lurb_borders[3]),
        )
    )


def expand_with_borders(rect, borders, up=0, bottom=0, left=0, right=0):
    return enable_borders(expand(rect, up, bottom, left, right), borders)


def lurb_conver_to_subrect_coords(lurb_convert_rect, lurb_subrect):
    return (
        lurb_convert_rect[0] - lurb_subrect[0],
        lurb_convert_rect[1] - lurb_subrect[1],
        lurb_convert_rect[2] - lurb_subrect[0],
        lurb_convert_rect[3] - lurb_subrect[1],
    )


def put_image(image, bg_image, x, y, mask=None):
    if mask is None:
        bg_image[y : (y + image.shape[0]), x : (x + image.shape[1])] = image
    else:
        bg_image[y : (y + image.shape[0]), x : (x + image.shape[1])] = np.asarray(
            image * mask
            + bg_image[y : (y + image.shape[0]), x : (x + image.shape[1])]
            * (1.0 - mask),
            dtype="uint8",
        )


def bgra_to_bgr_and_mask(bgra_image):
    mask = bgra_image[:, :, 3:] / 255.0
    mask = cv2.merge([mask, mask, mask])
    bgra_image = bgra_image[:, :, 0:3]

    return bgra_image, mask


def image_from_bytes(image_bytes):
    nparr = np.fromstring(image_bytes, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np
