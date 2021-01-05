import csv
import multiprocessing
import time
from contextlib import closing
from glob import glob
from math import ceil
from os.path import join, getsize
from pathlib import Path
from typing import List

import click
import cv2
import imageio
import imutils
import skimage
import yaml
import numpy as np
from skimage import img_as_ubyte

import thresholding
from options import TAOptions
from results import TAResult
from traits import traits, compute_medial_axis

MB_FACTOR = float(1 << 20)


def list_images(path: str, filetypes: List[str]):
    files = []
    for filetype in filetypes:
        files = files + sorted(glob(join(path, f"*.{filetype}")))
    return files


def write_result(path: str, result: TAResult):
    # YAML
    with open(join(path, "results.yml"), 'w') as file:
        yaml.dump(result, file, default_flow_style=False)

    # CSV
    with open(join(path, "results.csv"), 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(list(result.keys()))
        writer.writerow(list(result.values()))


def process(options: TAOptions) -> TAResult:
    result = TAResult(name=options.input_stem)
    output_prefix = join(options.output_directory, options.input_stem)
    print(f"Extracting traits from '{options.input_name}'")

    # read grayscale image
    gray_image = imageio.imread(options.input_file, as_gray=True)
    if len(gray_image) == 0:
        raise ValueError(f"Image is empty: {options.input_name}")

    # read color image
    color_image = imageio.imread(options.input_file, as_gray=False)
    if len(color_image) == 0:
        raise ValueError(f"Image is empty: {options.input_name}")

    # binary threshold
    masked_image = thresholding.binary_threshold(gray_image.astype(np.uint8))
    imageio.imwrite(f"{output_prefix}.mask.png", skimage.img_as_uint(masked_image))

    # erosion/dilation
    kernel = np.ones((7, 7), np.uint8)
    dilated_image = cv2.dilate(masked_image.copy(), kernel, iterations=2)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=2)
    imageio.imwrite(f"{output_prefix}.opened.png", dilated_image)

    # circle detection
    print(f"Finding circles")
    detected_circles = cv2.HoughCircles(cv2.blur(eroded_image.copy(), (5, 5)),
                                        cv2.HOUGH_GRADIENT, 1, 100, param1=40,
                                        param2=40, minRadius=50, maxRadius=350)

    circle_detection_copy = color_image.copy()
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(circle_detection_copy, (a, b), r, (0, 255, 0), 2)
            cv2.circle(circle_detection_copy, (a, b), 1, (0, 0, 255), 3)

        cv2.imwrite(f"{output_prefix}.circles.png", circle_detection_copy)

    # contour detection
    print(f"Finding contours")
    contours, hierarchy = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image = cv2.drawContours(color_image.copy(), contours, -1, (0, 255, 0), 2)

    contours_image = color_image.copy()
    min_area = 10000
    max_area = 200000
    filtered_counters = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True))
        area = cv2.contourArea(contour)
        rect = cv2.boundingRect(contour)
        if max_area > area > min_area:
            filtered_counters.append(contour)
            print(f"Contour area: {area}")
            print(f"Contour rect: {rect}")
            cv2.drawContours(contours_image, [contour], 0, (0, 255, 0), 3)

    print(f"Kept {len(filtered_counters)} of {len(contours)} total contours")
    cv2.imwrite(f"{output_prefix}.contours.png", contours_image)

    # edge detection
    print(f"Finding edges")
    edges_image = cv2.Canny(color_image, 100, 200)
    cv2.imwrite(f"{output_prefix}.edges.png", edges_image)

    # extract traits
    print(f"Extracting traits")
    result = {**result, **traits(color_image, output_prefix)}

    return result


@click.command()
@click.argument('input_file')
@click.option('--output_directory', required=False, type=str, default='')
def cli(input_file, output_directory):
    start = time.time()
    options = TAOptions(input_file, output_directory)
    result = process(options)

    print(f"Writing results to file")
    write_result(options.output_directory, result)

    duration = ceil((time.time() - start))
    print(f"Finished in {duration} seconds.")


if __name__ == '__main__':
    cli()
