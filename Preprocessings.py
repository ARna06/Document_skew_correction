import numpy as np
from scipy import ndimage


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def filters(img):
    K_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    K_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    I_x = ndimage.convolve(img, K_x)
    I_y = ndimage.convolve(img, K_y)

    G = np.hypot(I_x, I_y)
    G = G / G.max() * 255
    theta = np.arctan2(I_y, I_x)
    return (G, theta)


def suppress(img, theta):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def thresholding(img, low_Threshold_Ratio=0.05, high_Threshold_Ratio=0.09):
    high_Threshold = img.max() * high_Threshold_Ratio
    low_Threshold = high_Threshold * low_Threshold_Ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= high_Threshold)
    zeros_i, zeros_j = np.where(img < low_Threshold)

    weak_i, weak_j = np.where((img <= high_Threshold) & (img >= low_Threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)


def pixel_interpolation(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def Image_pre_processings(img):
    filtered, thetas = filters(rgb2gray(img))
    suppressed = suppress(filtered, thetas)
    thresholded, weak_pix, strong_pix = thresholding(suppressed)
    processed_image = pixel_interpolation(thresholded, weak_pix, strong_pix)

    return processed_image
