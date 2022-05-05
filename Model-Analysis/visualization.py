import numpy as np
import pandas as pd
import csv
import json
import argparse
import os
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as mcolors
from PIL import Image
from pathlib import Path

import sys
from utils import read_dataset, read_obj_vocab


def visualize_w_bb(img_path, info, obj_vocab, showbg=True):
	img = cv2.imread(img_path)

	cv2.imwrite("tmp_orig.png", img)
	box_color = (255,0,255)
	bg_color = (255,255,0)
	addbg_color = (0,255,255)
	
	objset = list(set(info['objects']))
	# print("set of objects: ", objset)

	for obj_idx in range(100):
		obj = info['objects'][obj_idx]
		box = info['bbox'][obj_idx]
		label = obj_vocab[obj]
		print("obj class idx: ", obj, "label: ", label)

		if (label == "background"):
			color = bg_color
		elif (label == "BACKGROUND"):
			color = addbg_color
			if (not showbg):
				continue
		else:
			color = box_color

		print("obj color index: ", objset.index(obj))
		print("color: ", color)

		cv2.rectangle(img,
						(int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
						color=color,
						thickness=2)
		cv2.rectangle(img,
						(int(box[0])-1, int(box[1])),
						(int(box[0])-1+12*len(label), int(box[1])-15),
						color=color,
						thickness=-1)
		cv2.putText(img, label,
						(int(box[0]), int(box[1]) - 3),
						cv2.FONT_HERSHEY_COMPLEX,
						0.5,
						(0, 0, 0),
						thickness=1)
	cv2.imwrite("tmp.png", img)

def visualize_img(img_path):
	img = cv2.imread(img_path)
	print(img_path)
	cv2.imwrite("tmp_orig.png", img)

def visualize_w_qa(data, img_path, info, subset_name):
	with open(img_path, "rb") as f:
		img_file = Image.open(f)
		img = img_file.convert("RGB")
	f.close()
	img_id = img_path.split("/")[-1]
	# save_path = "../../failure_cases/" + subset_name + "/" + str(data['question_id']) + "_" + img_id
	# print(save_path)
	print(data)

	# visualize_img(img_path)
	plt.text(0, -50, u'Question: {}'.format(data['question']), fontsize=10)
	plt.text(0, -20, u'Answer: {}'.format(data['answers']), fontsize=10)
	plt.imshow(img)
	plt.axis('off')
	# plt.savefig(save_path)
	plt.show()

# def process_gradebook(gradebook, args, dataset):
# 	idx = 0
# 	sample_gb = gradebook.iloc[idx]
# 	question_id = sample_gb['question_id']
# 	img_class = dataset[dataset['question_id']==question_id]['image_classes'].item()
# 	img_id = dataset[dataset['question_id']==question_id]['image_id'].item()
# 	set_name = dataset[dataset['question_id']==question_id]['set_name'].item()
# 	if (set_name=="test"):
# 		set_path = "test/"
# 		imgset_path = "test_images/"
# 	else:
# 		set_path = "train/"
# 		imgset_path = "train_images/"
# 	sample_info_path = args.DATAINFO_PATH + set_path + img_id + "_info.npy"
# 	img_path = args.IMAGE_PATH + imgset_path + img_id + ".jpg"
# 	visualize_img(img_path)

# def main(args):
# 	dataset_path = "../Data/textvqa/"
# 	trainset, valset, testset = read_dataset(dataset_path)
# 	obj_vocab = read_obj_vocab("../Data/objects_vocab.txt")

# 	### "all_zero": samples with 0.0 accuracy in each model
# 	### "zero_one": samples that get 1.0 accuracy in some models
# 	###                      and get 0.0 accuracy in others

# 	gradebook_path = "gradebooks/grade_book_" + args.GRADEBOOK + ".csv"
# 	gradebook = pd.read_csv(gradebook_path)
# 	print(args.GRADEBOOK, "\ttotal: ", len(gradebook))
# 	# print(gradebook.columns)

# 	process_gradebook(gradebook, args, valset)


# def parse_arg():
# 	parser = argparse.ArgumentParser()

# 	parser.add_argument('--GRADEBOOK', '-g',
# 						help='gradebook name',
# 						required=True)
# 	parser.add_argument('--DATAINFO_PATH', '-info',
# 						help='Data info path',
# 						default="../../data/")
# 	parser.add_argument('--IMAGE_PATH', '-i',
# 						help='Data info path',
# 						default="../../data/textvqa/")
# 	args = parser.parse_args()
# 	return args


# if __name__ == '__main__':
	args = parse_arg()
	main(args)