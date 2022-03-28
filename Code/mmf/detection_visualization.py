import argparse
import pickle
import random
from pathlib import Path
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def drawRect(coord: list):
    sx, sy, ex, ey = coord[0], coord[1], coord[2], coord[3]
    coord = [[sx, sy], [sx, ey], [ex, ey], [ex, sy], [sx, sy]]
    # coord.append(coord[0]) #repeat the first point to create a 'closed loop'
    xs, ys = zip(*coord)  # create lists of x and y values
    plt.plot(xs, ys)


def drawDots(coord):
    x_dots, y_dots = zip(*coord)
    plt.scatter(x_dots, y_dots, s=5)


def get_coord(bbox):
    sx,sy,ex, ey = bbox
    return [sx, sy, ex, ey]


def default_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def object_visualization(img_path, pickle_path):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    que, bbox, labels, image_name = data
    image_name = image_name.split("/")[-1]
    img = default_loader(os.path.join(img_path, image_name))
    img_width, img_height = img.size
    print("image_shape: ", img.size)
    print("que text: ", que)
    for c, text in zip(bbox, labels):
        if text == "background":
            continue
        print("coord: ", c, "text: ", text)
        plt.text(c[0], c[1], u'{}'.format(text), fontsize=10)
        drawRect(c)
    plt.imshow(img)
    plt.show()
    plt.savefig(os.path.join(img_path, image_name+"_ocr_visualized.jpg"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Detected Object visualization.')
    parser.add_argument('--input_image_path', type=Path, required=True, help='Path to input.')
    parser.add_argument('--input_pkl_path', type=Path, required=True, help='Path to input.')
    args = parser.parse_args()
    object_visualization(args.input_image_path, args.input_pkl_path)
    # ocr_with_qa_visualization(args.input_question_path, args.input_image_path, args.input_ocr_path)
