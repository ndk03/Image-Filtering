import argparse
import json
import os

import utils
from task1 import *


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task2.jpg",
        help="path to the image")
    parser.add_argument(
        "--template-path",
        type=str,
        default="./data/proj1-task2-template.jpg",
        help="path to the template"
    )
    parser.add_argument(
        "--result-saving-path",
        dest="rs_path",
        type=str,
        default="./results/task2.json",
        help="path to file which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args

def norm_xcorr2d(patch, template):
    """Computes the NCC value between a image patch and a template.

    The image patch and the template are of the same size. The formula used to compute the NCC value is:
    sum_{i,j}(x_{i,j} - x^{m}_{i,j})(y_{i,j} - y^{m}_{i,j}) / (sum_{i,j}(x_{i,j} - x^{m}_{i,j}) ** 2 * sum_{i,j}(y_{i,j} - y^{m}_{i,j})) ** 0.5
    This equation is the one shown in Prof. Yuan's ppt.

    Args:
        patch: nested list (int), image patch.
        template: nested list (int), template.

    Returns:
        value (float): the NCC value between a image patch and a template.
    """
    flipped_template = template
    #print(flipped_template.shape)
    #print(patch.shape)
    rows = 0
    cols = 0
    mean_patch = 0
    mean_template = 0
    
    #calculating the mean of patch
    for i in range(0,len(patch)):
        for j in range(0,len(patch[1])):
            mean_patch = mean_patch + patch[i][j]
    mean_patch = mean_patch/(len(patch)*len(patch[1]))
    
    #calculating the mean of template
    for i in range(0,len(flipped_template)):
        for j in range(0,len(flipped_template[1])):
            mean_template = mean_template + flipped_template[i][j]
    mean_template = mean_template/(len(flipped_template)*len(flipped_template[1]))
    
    numerator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0
    
    for i in range(0,len(patch)):
        for j in range(0,len(patch[1])):
            numerator = numerator + (flipped_template[i][j]-mean_template)*(patch[i][j] - mean_patch)
            denominator1 = denominator1 + (flipped_template[i][j]-mean_template)**2    
            denominator2 = denominator2 + (patch[i][j]-mean_patch)**2
    denominator = (denominator1*denominator2)**(1/2)
    
    return(numerator/denominator)
    #raise NotImplementedError
        
def match(img, template):
    """Locates the template, i.e., a image patch, in a large image using template matching techniques, i.e., NCC.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        x (int): row that the character appears (starts from 0).
        y (int): column that the character appears (starts from 0).
        max_value (float): maximum NCC value.
    """
    position = []
    ncc = []
    for i in range(0,len(img)-len(template)):
        for j in range(0,len(img[1])-len(template[1])):
            patch = utils.crop(img,i,i+len(template),j,j+len(template[0]))
            """for ki in range(0,len(template)):
                new_row = []
                for kj in range(0,len(template[1])):
                    new_row.append(img[i+ki][j+kj])
                patch.append(new_row)"""
            ncc.append(norm_xcorr2d(patch,template))
            position.append([i,j])
    
    max_index = 0
    max = ncc[0]
    for i in range(1,len(ncc)):
        if(ncc[i]>max):
            max = ncc[i]
            max_index = i
    
    x = position[max_index][0]
    y = position[max_index][1]
    return x,y,max

def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    # template = utils.crop(img, xmin=10, xmax=30, ymin=10, ymax=30)
    # template = np.asarray(template, dtype=np.uint8)
    # cv2.imwrite("./data/proj1-task2-template.jpg", template)
    template = read_image(args.template_path)

    x, y, max_value = match(img, template)
    with open(args.rs_path, "w") as file:
        json.dump({"x": x, "y": y, "value": max_value}, file)


if __name__ == "__main__":
    main()
