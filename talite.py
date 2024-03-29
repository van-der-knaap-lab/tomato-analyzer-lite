import csv
import time
from glob import glob
from math import ceil
from os.path import join
from typing import List

import click
import cv2
import imageio
import numpy as np
import skimage
import yaml
from plantcv import plantcv as pcv

import thresholding
from options import TAOptions
from results import TAResult

MB_FACTOR = float(1 << 20)


def list_images(path: str, filetypes: List[str]):
    files = []
    for filetype in filetypes:
        files = files + sorted(glob(join(path, f"*.{filetype}")))
    return files


def write_results(options: TAOptions, results: List[TAResult]):
    # YAML
    with open(join(options.output_directory, f"{options.input_stem}.results.yml"), 'w') as file:
        yaml.dump({'features': results}, file, default_flow_style=False)

    # CSV
    with open(join(options.output_directory, f"{options.input_stem}.results.csv"), 'w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if len(results) != 0:
            writer.writerow(list(results[0].keys()))
        for result in results:
            writer.writerow(list(result.values()))


def process(options: TAOptions) -> List[TAResult]:
    results = []
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

    # kernel = np.ones((7, 7), np.uint8)
    # dilated_image = cv2.dilate(blurred_image, kernel, iterations=1)
    # eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
    # imageio.imwrite(f"{output_prefix}.dilated.png", dilated_image)
    # imageio.imwrite(f"{output_prefix}.eroded.png", eroded_image)

    # binary threshold
    masked_image = thresholding.binary_threshold(gray_image.astype(np.uint8))
    imageio.imwrite(f"{output_prefix}.mask.png", skimage.img_as_uint(masked_image))

    # closing (dilation/erosion)
    kernel = np.ones((7, 7), np.uint8)
    dilated_image = cv2.dilate(masked_image, kernel, iterations=1)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
    imageio.imwrite(f"{output_prefix}.dilated.png", dilated_image)
    imageio.imwrite(f"{output_prefix}.eroded.png", eroded_image)

    # circle detection
    # print(f"Finding circles")
    # detected_circles = cv2.HoughCircles(cv2.blur(eroded_image.copy(), (5, 5)),
    #                                     cv2.HOUGH_GRADIENT, 1, 40, param1=40,
    #                                     param2=39, minRadius=20, maxRadius=500)

    # circle_detection_copy = color_image.copy()
    # if detected_circles is not None:
    #     detected_circles = np.uint16(np.around(detected_circles))
    #     for pt in detected_circles[0, :]:
    #         a, b, r = pt[0], pt[1], pt[2]
    #         cv2.circle(circle_detection_copy, (a, b), r, (0, 255, 0), 2)
    #         cv2.circle(circle_detection_copy, (a, b), 1, (0, 0, 255), 3)

    #     cv2.imwrite(f"{output_prefix}.circles.png", circle_detection_copy)

    # contour detection
    # TODO exclude shapes which are square or rectangular within a certain error range
    # TODO compute and return area/curvature/solidity for each contour
    print(f"Finding contours")
    closed_image = cv2.morphologyEx(dilated_image.copy(), cv2.MORPH_CLOSE, kernel)
    # contours, hierarchy = cv2.findContours(closed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_image, contours, hierarchy = cv2.findContours(closed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # opencv 3 compat
    contours_image = color_image.copy()
    min_area = 10000
    max_area = 200000
    filtered_contours = []
    for i, contour in enumerate(contours):
        cnt = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
        bounding_rect = cv2.boundingRect(cnt)
        (x, y, w, h) = bounding_rect
        min_rect = cv2.minAreaRect(cnt)
        area = cv2.contourArea(contour)
        rect_area = w * h
        if max_area > area > min_area and abs(area - rect_area) > 0.3:
            filtered_contours.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })

            # draw and label contours
            cv2.drawContours(contours_image, [contour], 0, (0, 255, 0), 3)
            cv2.putText(contours_image, str(i), (x + 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # draw min bounding box
            box = np.int0(cv2.boxPoints(min_rect))
            cv2.drawContours(contours_image, [box], 0, (0, 0, 255), 2)

            results.append(TAResult(
                id=str(i),
                area=area,
                solidity=min(round(area / rect_area, 4), 1),
                max_height=h,
                max_width=w))

            # find objects (PlantCV)
            id_objects, obj_hierarchy = pcv.find_objects(img=closed_image, mask=closed_image)
            roi1, roi_hierarchy = pcv.roi.rectangle(img=closed_image, x=x, y=y, h=h, w=w)
            roi_objects, hierarchy3, kept_mask, obj_area = pcv.roi_objects(img=closed_image, roi_contour=roi1,
                                                                           roi_hierarchy=roi_hierarchy,
                                                                           object_contour=id_objects,
                                                                           obj_hierarchy=obj_hierarchy,
                                                                           roi_type='partial')
            obj, mask = pcv.object_composition(img=closed_image, contours=roi_objects, hierarchy=hierarchy3)
            analysis_image = pcv.analyze_object(img=closed_image, obj=obj, mask=mask, label="default")
            pcv.outputs.save_results(filename=f"{output_prefix}.analysis.{i}.csv", outformat='csv')
            cv2.imwrite(f"{output_prefix}.analysis.{i}.png", skimage.img_as_uint(analysis_image))

    print(f"Kept {len(filtered_contours)} of {len(contours)} total contours")
    cv2.imwrite(f"{output_prefix}.contours.png", contours_image)

    # edge detection
    print(f"Finding edges")
    edges_image = cv2.Canny(color_image, 100, 200)
    cv2.imwrite(f"{output_prefix}.edges.png", edges_image)

    return results


@click.command()
@click.argument('input_file')
@click.option('-o', '--output_directory', required=False, type=str, default='')
def cli(input_file, output_directory):
    start = time.time()
    options = TAOptions(input_file, output_directory)
    results = process(options)

    print(f"Writing results to file")
    write_results(options, results)

    duration = ceil((time.time() - start))
    print(f"Finished in {duration} seconds.")


if __name__ == '__main__':
    cli()
