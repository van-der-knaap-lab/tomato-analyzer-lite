import warnings
from collections import Counter
from os.path import join, getsize
from pathlib import Path

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.spatial import distance as dist
from skimage import img_as_float, img_as_ubyte, img_as_bool
from skimage.color import rgb2lab, deltaE_cie76
from skimage.feature import peak_local_max
from skimage.morphology import watershed, medial_axis
from sklearn.cluster import KMeans

from options import TAOptions
from results import TAResult

warnings.filterwarnings("ignore")

MB_FACTOR = float(1 << 20)


class ComputeCurvature:

    def __init__(self, x, y):
        """ Initialize some variables """
        self.xc = 0  # X-coordinate of circle center
        self.yc = 0  # Y-coordinate of circle center
        self.r = 0  # Radius of the circle
        self.xx = np.array([])  # Data points
        self.yy = np.array([])  # Data points
        self.x = x  # X-coordinate of circle center
        self.y = y  # Y-coordinate of circle center

    def calc_r(self, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((self.xx - xc) ** 2 + (self.yy - yc) ** 2)

    def f(self, c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        ri = self.calc_r(*c)
        return ri - ri.mean()

    def df(self, c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df_dc = np.empty((len(c), self.x.size))

        ri = self.calc_r(xc, yc)
        df_dc[0] = (xc - self.x) / ri  # dR/dxc
        df_dc[1] = (yc - self.y) / ri  # dR/dyc
        df_dc = df_dc - df_dc.mean(axis=1)[:, np.newaxis]
        return df_dc

    def fit(self, xx, yy):
        self.xx = xx
        self.yy = yy
        center_estimate = np.r_[np.mean(xx), np.mean(yy)]
        center = optimize.leastsq(self.f, center_estimate, Dfun=self.df, col_deriv=True)[0]

        self.xc, self.yc = center
        ri = self.calc_r(*center)
        self.r = ri.mean()

        return 1 / self.r  # Return the curvature


def color_cluster_seg(image):
    (width, height, n_channel) = image.shape

    # flatten the 2D image array into an MxN feature vector
    # M is the number of pixels and N is the dimension (number of channels)
    reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, n_init=40, max_iter=500).fit(reshaped)
    pred_label = kmeans.labels_

    # reshape result back into a 2D array
    # each element represents the corresponding pixel's cluster index (0 to K - 1).
    clustering = np.reshape(np.array(pred_label, dtype=np.uint8), (image.shape[0], image.shape[1]))

    # sort cluster labels in order of frequency with which they occur
    # sortedLabels = sorted([n for n in range(args_num_clusters)], key=lambda x: -np.sum(clustering == x))

    kmeansImage = np.zeros(image.shape[:2], dtype=np.uint8)
    for i, label in enumerate(clustering):
        kmeansImage[clustering == label] = int(255 / (num_clusters - 1)) * i

    ret, thresh = cv2.threshold(kmeansImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    component_count, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    sizes = stats[1:, -1]
    component_count = component_count - 1
    min_size = 150
    threshold_image = np.zeros([width, height], dtype=np.uint8)

    # for every component in the image, keep it only if it's above min_size
    for i in range(0, component_count):
        if sizes[i] >= min_size:
            threshold_image[output == i + 1] = 255

    return threshold_image


def compute_medial_axis(image):
    image_medial_axis = medial_axis(img_as_bool(img_as_float(image)))
    return image_medial_axis


def apply_watershed(image, min_distance_value):
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(image)
    localMax = peak_local_max(D, indices=False, min_distance=min_distance_value, labels=image)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    components = watershed(-D, markers, mask=image)

    print(f"{len(np.unique(components)) - 1} unique components found")
    return components


def external_contours(orig, thresh):
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_height, img_width, img_channels = orig.shape
    index = 1

    for contour in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        if w > img_width * 0.1 and h > img_height * 0.1:
            index += 1
            area = cv2.contourArea(contour)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area

            extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
            extRight = tuple(contour[contour[:, :, 0].argmax()][0])
            extTop = tuple(contour[contour[:, :, 1].argmin()][0])
            extBot = tuple(contour[contour[:, :, 1].argmax()][0])

            max_width = dist.euclidean(extLeft, extRight)
            max_height = dist.euclidean(extTop, extBot)

            if max_width > max_height:
                trait_img = cv2.line(orig, extLeft, extRight, (0, 255, 0), 2)
            else:
                trait_img = cv2.line(orig, extTop, extBot, (0, 255, 0), 2)

    return trait_img, area, solidity, w, h


def compute_curvature(image, labels):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    curv_sum = 0.0
    count = 0

    for index, label in enumerate(np.unique(labels), start=1):
        # if the label is zero, we are examining the 'background' so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)

        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)

        # optional to "delete" the small contours
        if len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            label_trait = cv2.ellipse(image, ellipse, (0, 255, 0), 2)

            c_np = np.vstack(c).squeeze()
            count += 1

            x = c_np[:, 0]
            y = c_np[:, 1]

            comp_curv = ComputeCurvature(x, y)
            curvature = comp_curv.fit(x, y)

            curv_sum = curv_sum + curvature
        else:
            label_trait = cv2.drawContours(image, [c], -1, (0, 0, 255), 2)

    return curv_sum / count, label_trait


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def color_kmeans(image, num_clusters):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = np.float32(rgb_image.reshape((-1, 3)))
    stopping_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, (centers) = cv2.kmeans(
        pixel_values,
        num_clusters,
        None,
        stopping_criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS)

    return rgb_image, compactness, labels.flatten(), np.uint8(centers)


def color_region(image, num_clusters):
    rgb_image, compactness, flat_labels, centers = color_kmeans(image, num_clusters)
    masked_image = np.zeros_like(rgb_image)
    masked_image = masked_image.reshape((-1, 3))
    color_conversion = interp1d([0, 1], [0, 255])

    for cluster in range(num_clusters):
        print(f"Processing cluster {cluster}")
        masked_image[flat_labels == cluster] = centers[cluster]
        masked_image_rp = masked_image.reshape(rgb_image.shape)
        gray_image = cv2.cvtColor(masked_image_rp, cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        cluster_images = []

        if not contours or len(contours) == 0:
            print("No contours found")
        else:
            for (i, contour) in enumerate(contours):
                result = cv2.drawContours(masked_image_rp, contour, -1, color_conversion(np.random.random(3)), 2)

            result[result == 0] = 255
            bgr_result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cluster_images.append(bgr_result)

    counts = dict(sorted(Counter(flat_labels).items()))
    center_colors = centers
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    index_bkg = [index for index in range(len(hex_colors)) if hex_colors[index] == '#000000']

    del hex_colors[index_bkg[0]]
    del rgb_colors[index_bkg[0]]

    delete = [key for key in counts if key == index_bkg[0]]
    for key in delete:
        del counts[key]

    return rgb_colors, cluster_images, counts.values(), hex_colors


def traits(image: np.ndarray, output_prefix: str) -> TAResult:
    image_copy = image.copy()

    # labels = apply_watershed(image_copy, 5)
    # label_hue = np.uint8(128 * labels / np.max(labels))
    # blank_ch = 255 * np.ones_like(label_hue)
    # labeled_image = cv2.merge([label_hue, blank_ch, blank_ch])
    # labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_HSV2BGR)
    # labeled_image[label_hue == 0] = 0  # set background label to black
    # cv2.imwrite(f"{output_prefix}.lab.png", labeled_image)

    # (curvature, curvature_image) = compute_curvature(image_copy, labels)
    # cv2.imwrite(f"{output_prefix}.cur.png", curvature_image)

    # segmented_image = color_cluster_seg(image_copy)
    # cv2.imwrite(f"{output_prefix}.seg.png", segmented_image)

    # (contour_image, area, solidity, max_width, max_height) = external_contours(image.copy(), segmented_image)
    # cv2.imwrite(f"{output_prefix}.con.png", contour_image)

    return TAResult(
        # area=area,
        # solidity=solidity,
        # max_width=max_width,
        # max_height=max_height,
        # mean_curvature=curvature
    )
