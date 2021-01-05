import numpy as np
import cv2


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    image = cv2.createCLAHE(clipLimit=3).apply(image)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image


def binary_threshold(image: np.ndarray):
    if len(np.unique(image)) <= 2:
        print('Binary input detected, skipping threshold')
        idx1 = np.where(image == np.unique(image)[0])
        idx2 = np.where(image == np.unique(image)[1])
        image[idx1] = False
        image[idx2] = True
    else:
        print('Applying binary threshold')
        image = otsu_threshold(image)

    w, h = np.shape(image)
    image[np.where(image == np.unique(image)[0])] = 0       # set to black
    image[np.where(image == np.unique(image)[1])] = 255     # set to white
    image[0, :] = 0
    image[:, 0] = 0
    image[w - 1, :] = 0
    image[:, h - 1] = 0

    return image
