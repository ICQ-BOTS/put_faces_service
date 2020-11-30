from keras.models import model_from_json
import os
import cv2
import numpy as np
import random


img_rows = 96
img_cols = 96
model_name_load = "weight_unet_96"
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


def test_NN():
    # 1. Грузим модель НС
    with open(os.path.join("./models", model_name_load + ".json"), "r") as f:
        model_segment = model_from_json(f.read())
    model_segment.load_weights(os.path.join("./models", model_name_load + ".h5"))

    # 2. Грузим человека
    frame_src = cv2.imread("test.jpg")

    # prepare frame for NN
    # frame_src = cropImageToSquare(frame_src)
    frame = cv2.resize(frame_src, (img_rows, img_cols))
    frame = frame / 127.5 - 1

    [h, w] = frame_src.shape[:2]
    bg_image = cv2.imread(bg_path)
    bg_image = cropImageToSquare(bg_image)
    bg_image = cv2.resize(bg_image, (h, w))

    # predict
    inputImage = np.asarray([frame], dtype="float32")
    # Находим человека и вырезаем его
    mask = model_segment.predict(inputImage)[0].squeeze()
    mask = np.asarray(mask * 255, dtype="uint8")
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)

    # filter mask
    # Убираем шум морфологическим открытием
    mask = filterMorph(mask)
    # Находим самый большой контур на изображении и создаём маску по нему
    mask = filterConetectedComponents(mask)

    # replace bg
    mask_src = cv2.resize(mask, (h, w))
    # mask_src = erode(mask_src)
    mask_src = smoothContour(mask_src)
    mask_fg = cv2.blur(mask_src, (7, 7))
    mask_fg = cv2.merge([mask_fg, mask_fg, mask_fg])
    mask_bg = 255 - mask_fg
    mask_fg = mask_fg / 255
    mask_bg = mask_bg / 255
    mix_image = bg_image * mask_bg + frame_src * mask_fg
    mix_image = np.asarray(mix_image, dtype="uint8")

    # Display the resulting frame
    cv2.imshow("frame", frame_src)
    cv2.imshow("mask", mask_src)
    cv2.imshow("mix", mix_image)
    cv2.imwrite("result.jpg", mix_image)
    cv2.waitKey(-1)


test_NN()
exit()


def resize_proportional(image, height=None, width=None, long_side=None):
    if long_side != None:
        if image.shape[0] > image.shape[1]:
            height = long_side
        else:
            width = long_side
    if height != None:
        return cv2.resize(
            image, (int(image.shape[1] / image.shape[0] * height), height)
        )
    if width != None:
        print(width / (image.shape[1] / image.shape[0]))
        return cv2.resize(
            image, (width, int(width / (image.shape[1] / image.shape[0])))
        )


def find_put_areas(image):
    if image.shape[1] > 400:
        image = resize_proportional(image, width=400)
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw_image = cv2.equalizeHist(bw_image)
    blurred_image = cv2.GaussianBlur(bw_image, (3, 3), 1)
    canny_image = cv2.Canny(blurred_image, 190, 255, 1)
    dilated_image = cv2.dilate(
        canny_image, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=1
    )
    eroded_image = cv2.erode(
        dilated_image,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=4,
    )
    blurred_eroded = cv2.GaussianBlur(eroded_image, (7, 7), 4)
    threshed_image = cv2.threshold(blurred_eroded, 127, 255, cv2.THRESH_BINARY)[1]

    # cv2.imshow('Original', image)
    # cv2.imshow('B&W', bw_image)
    # cv2.imshow('Blured image', blurred_image)
    # cv2.imshow('Canny', canny_image)
    # cv2.imshow('Eroded', eroded_image)
    # cv2.imshow('B&W 1', cv2.Canny(bw_image, 210, 255, 1))
    # cv2.imshow('B&W 2', cv2.Canny(bw_image, 120, 200, 1))
    # cv2.imshow('B&W 3', cv2.Canny(bw_image, 180, 200, 1))

    cv2.imshow("Canny", canny_image)
    cv2.imshow("Dilate", eroded_image)
    cv2.imshow("No arts 1", blurred_eroded)
    cv2.imshow("No arts 2", threshed_image)

    # sift = cv2.SIFT_create()
    # kp = sift.detect(bw_image, None)
    return
    orb = cv2.ORB_create(400)
    key_points, des = orb.detectAndCompute(bw_image, None)
    kp_image = cv2.drawKeypoints(bw_image, key_points, bw_image)

    kp_coords_matrix = np.zeros((len(key_points), 2), np.float32)
    counter = 0
    for key_point in key_points:
        # print(key_point.pt)
        kp_coords_matrix[counter] = key_point.pt
        counter += 1

    cv2.imshow("Key points", kp_image)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret, label, center = cv2.kmeans(
        kp_coords_matrix, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    index = 0

    clusters_colors = []
    for index in range(K):
        clusters_colors.append(
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        )

    clusters_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
    for index in range(kp_coords_matrix.shape[0]):
        # print(f'{kp_coords_matrix[index]}: {label[index][0]} ({tuple(clusters_colors[label[index][0]])})')
        key_point = key_points[index].pt
        cv2.circle(
            clusters_image,
            (int(key_point[0]), int(key_point[1])),
            4,
            tuple(clusters_colors[label[index][0]]),
            -1,
        )

    cv2.imshow("Clusters", clusters_image)

    # Now convert back into uint8, and make original image
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res2 = res.reshape((img.shape))

    # cv2.imshow('Erode', eroded_image)
    """
    countours_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
    countours, hierarchy = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(countours_image, countours, -1, (0, 255, 0), 2)
    
    #cv2.imshow('Countours', countours_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    opening = cv2.morphologyEx(eroded_image, cv2.MORPH_CLOSE, kernel)
    #closing = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow('Morph 1', eroded_image)
    cv2.imshow('Morph 2', opening)
    #cv2.imshow('Morph opening', closing)
    """
    # big_countours = []
    # for countour in countours:
    # area = cv2.counoutArea(countour)
    # peri = cv2.arcLength(countour, True)
    # approx = cv2.approxPolyDP(countour, 0.02 * peri, True)


# find_put_areas(cv2.imread(bg_path))


def reduce_bw_image(image, colors_count):
    step = 256 / colors_count
    # addition = np.ones((image.shape[0], image.shape[1]), np.uint8) * step

    result = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    for index in range(colors_count):
        ret, thresh_image = cv2.threshold(
            image, (index + 1) * step, index * step, cv2.THRESH_BINARY
        )
        result = np.maximum(result, thresh_image)
    return result


def find_circles(image):
    if image.shape[1] > 400:
        image = resize_proportional(image, width=400)
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("1", image)
    cv2.imshow("2", bw_image)

    blurred_image = cv2.GaussianBlur(bw_image, (15, 15), 2)

    # cv2.imshow('3', blurred_image)

    reduced_image = reduce_bw_image(blurred_image, 8)
    cv2.imshow("4", reduced_image)

    # circles = cv2.HoughCircles(reduced_image, cv2.cv.CV_HOUGH_GRADIENT, 1, 20, param1=30, param2=15, minRadius=0, maxRadius=0)
    canny_image = cv2.Canny(reduced_image, 127, 220, 1)
    cv2.imshow("5", canny_image)


find_circles(cv2.imread(bg_path))
cv2.waitKey(-1)