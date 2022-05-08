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

import sys
from utils import read_dataset, read_obj_vocab
from visualization import visualize_w_bb

def main():
	dataset_path = "../Data/textvqa/"
	
	parser = argparse.ArgumentParser()

	parser.add_argument('--DATAINFO_PATH', '-info',
						help='Data info path',
						default="../../data/")
	parser.add_argument('--OBJ_VOCAB_PATH', '-vocab',
						help='Object vocab file path',
						default="../../data/")
	args = parser.parse_args()

	trainset, valset, testset = read_dataset(dataset_path)
	obj_vocab = read_obj_vocab(args.OBJ_VOCAB_PATH+"objects_vocab.txt")
	filter_obj_questions(valset, args.DATAINFO_PATH, dataset_path, obj_vocab)


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
	downloaded = 0
	##### check each data sample #####
	for idx in range(len(dataset)):
		datasample = dataset.iloc[idx]
		imgsample = datasample['image_id']
		
		info_path = datainfo_root + data_sourceset + "/" + imgsample + "_info.npy"
		if os.path.exists(info_path):
			info = np.load(info_path, encoding = "latin1", allow_pickle=True).item()
			downloaded += 1

			# img_path = "../../data/textvqa/" + data_sourceset + "_images/" + imgsample + ".jpg"
			# print(img_path)
			# visualize_w_bb(img_path, info, obj_vocab)

			objects = info['objects']
			# if (bgindex in objects):
			# 	visualize_w_bb(img_path, info, obj_vocab)

			vocabs = set([obj_vocab[idx].lower() for idx in objects])
			for v in vocabs:
				if v in datasample['question']:
					subset_str += str(datasample['question_id']) + ","
					subset_index_str += str(idx) + ","
					subset_num += 1
					break

	print("Downloaded questions: ", downloaded)
	print("There are ", subset_num, " questions containing object vocabs.")
	subset_str = subset_str[:-1]
	subset_index_str = subset_index_str[:-1]

	with open("subsets/"+dataset['set_name'][0]+"_obj_related_question.txt", 'w') as f:
		f.write(subset_str)
	f.close()

	subset_str_text = ""
	for i in subset_index_str.split(","):
		subset_str_text += str(dataset.iloc[int(i)]['question_id']) + "\t" + str(dataset.iloc[int(i)]['question']) + "\n"
	
	# print(subset_str_text)
	with open("subsets/"+dataset['set_name'][0]+"_obj_related_question_text.txt", 'w') as f:
		f.write(subset_str_text)
	f.close()



if __name__ == '__main__':
	main()



