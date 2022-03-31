import numpy as np
import json
import csv
import pandas as pd
import argparse
import os

import random
from pathlib import Path
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as mcolors
from PIL import Image

def main():
	dataset_path = "./data/textvqa/"
	datainfo_path = "./data/"

	trainset, valset, testset = read_dataset(dataset_path)
	obj_vocab = read_obj_vocab(datainfo_path)
	filter_obj_questions(valset, datainfo_path, dataset_path, obj_vocab)

def read_obj_vocab(datainfo_root):
	with open(datainfo_root + "objects_vocab.txt", 'r') as f:
		obj_vocab = f.read().splitlines()
	obj_vocab = ["BACKGROUND"] + obj_vocab
	bgindex = obj_vocab.index("background")
	print("object volab len:", len(obj_vocab))
	print("background index:", bgindex)
	return obj_vocab


def colors_list(NUM_COLORS):
    def get_cmap(N):
        color_norm = mcolors.Normalize(0, N)
        return cmx.ScalarMappable(color_norm, 'hsv').to_rgba
    cmap = get_cmap(NUM_COLORS)
    colors = [cmap(float(i)) for i in range(NUM_COLORS)]
    return colors


def visualize_w_bb(img_path, info, obj_vocab):
	img = cv2.imread(img_path)
	cv2.imwrite("tmp_orig.png", img)
	box_color = (255,0,255)
	bg_color = (255,255,0)
	addbg_color = (0,255,255)
	
	showbg = True
	objset = list(set(info['objects']))
	print("set of objects: ", objset)
	colors = colors_list(len(objset))
	##### visualize #####
	for obj_idx in range(100):
		obj = info['objects'][obj_idx]
		box = info['bbox'][obj_idx]
		label = obj_vocab[obj]
		# print("obj class idx: ", obj, "label: ", label)

		# if (label == "background"):
		# 	color = bg_color
		# elif (label == "BACKGROUND"):
		# 	color = addbg_color
		# else:
		# 	color = box_color

		color = colors[objset.index(obj)]
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
	# cv2.imshow("temp", img)
	# cv2.waitKey()
	# cv2.destroyAllWindows()


def filter_obj_questions(dataset, datainfo_root, dataset_path, obj_vocab):
	##### process dataset #####
	#['objects', 'num_boxes', 'image_width', 'image_height', 'bbox', 'conf']
	if (dataset['set_name'][0] == "test"):
		# datainfo_path = datainfo_root + "test/"
		data_sourceset = "test"
	else:
		# datainfo_path = datainfo_root + "train/"
		data_sourceset = "train"

	subset_str = ""
	subset_index_str = ""
	print("dataset total length = ", len(dataset))
	subset_num = 0


	bgindex = obj_vocab.index("background")

	##### check each data sample #####
	for idx in range(len(dataset)):
	# for idx in range(2):
		datasample = dataset.iloc[idx]
		imgsample = datasample['image_id']
		
		info_path = datainfo_root + data_sourceset + "/" + imgsample + "_info.npy"
		if os.path.exists(info_path):
			info = np.load(info_path, encoding = "latin1", allow_pickle=True).item()

			img_path = dataset_path + data_sourceset + "_images/" + imgsample + ".jpg"
			# visualize_w_bb(img_path, info, obj_vocab)

			objects = info['objects']
			if (bgindex in objects):
				visualize_w_bb(img_path, info, obj_vocab)
				# print("yes")

			vocabs = set([obj_vocab[idx].lower() for idx in objects])
			for v in vocabs:
				if v in datasample['question']:
					subset_str += str(datasample['question_id']) + ","
					subset_index_str += str(idx) + ","
					subset_num += 1
					break

	print("There are ", subset_num, " questions containing object vocabs.")
	# subset_str = subset_str[:-1]
	# subset_index_str = subset_index_str[:-1]

	# with open(dataset['set_name'][0]+"_obj_related_question.txt", 'w') as f:
	# 	f.write(subset_str)
	# f.close()

	# subset_str_text = ""
	# for i in subset_index_str.split(","):
	# 	subset_str_text += str(dataset.iloc[int(i)]['question_id']) + "\t" + str(dataset.iloc[int(i)]['question']) + "\n"
	
	# # print(subset_str_text)
	# with open(dataset['set_name'][0]+"_obj_related_question_text.txt", 'w') as f:
	# 	f.write(subset_str_text)
	# f.close()


def read_dataset(dataset_path):
	with open(dataset_path + 'TextVQA_0.5.1_val.json') as val, \
		open(dataset_path + 'TextVQA_0.5.1_train.json') as train, \
		open(dataset_path + 'TextVQA_0.5.1_test.json') as test:
		val_list = json.load(val)
		train_list = json.load(train)
		test_list = json.load(test)

	val.close()
	train.close()
	test.close()

	val_data = pd.json_normalize(val_list, record_path='data')
	train_data = pd.json_normalize(train_list, record_path='data')
	test_data = pd.json_normalize(test_list, record_path='data')


	"""
	dataset columns:
		['question', 'image_id', 'image_classes', 'flickr_original_url',
		'flickr_300k_url', 'image_width', 'image_height', 'answers',
		'question_tokens', 'question_id', 'set_name']
	"""

	val_text = val_data[['question_id', 'question', 'image_id', 'set_name']]
	train_text = train_data[['question_id', 'question', 'image_id', 'set_name']]
	test_text = test_data[['question_id', 'question', 'image_id', 'set_name']]

	return train_text, val_text, test_text


if __name__ == '__main__':
	main()



