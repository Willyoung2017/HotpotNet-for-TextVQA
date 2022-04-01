import pandas as pd
import numpy as np
import pickle
import argparse
import os
import random
from pathlib import Path
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from utils import read_dataset, read_obj_vocab
from visualization import visualize_w_bb, visualize_img



def main(args):
	gradebook_path = "gradebooks/grade_book_" + args.GRADEBOOK + "_with_pred.csv"
	gradebook = pd.read_csv(gradebook_path)
	models = {'multimodal-baselines':
				['hotpot-lorra',
				'hotpot-without-mmt',
				'hotpot-without-object-label',
				'hotpot'],
			'unimodal-baselines':
				['ocr', 'question', 'question'],
			'competitive-baselines':
				['lorra', 'm4c', 'tap']}
	with open("model_preds.pickle", "rb") as f:
		model_preds_all = pickle.load(f)
	f.close()

	# subset = pd.read_csv("gradebooks/grade_book_" + args.SUBSETS + "_with_pred.csv")

	dataset_path = "../Data/textvqa/"
	trainset, valset, testset = read_dataset(dataset_path)
	obj_vocab = read_obj_vocab("../Data/objects_vocab.txt")
	df = gradebook.merge(valset[['question_id', 'image_classes', 'image_id', 'set_name']], on='question_id')
	

	show_gradebook(df, args, valset)


def visualize_w_pred(data, img_path, info_path, subset):
	with open(img_path, "rb") as f:
		img_file = Image.open(f)
		img = img_file.convert("RGB")
	f.close()
	img_id = img_path.split("/")[-1]
	save_path = "../../failure_cases/" + subset + "/" + str(data['question_id']) + "_" + img_id
	print(save_path)
	print(data)

	# visualize_img(img_path)
	plt.text(0, -50, u'Question: {}'.format(data['question']), fontsize=10)
	plt.text(0, -20, u'Answer: {}'.format(data['answers']), fontsize=10)
	plt.imshow(img)
	plt.axis('off')
	plt.savefig(save_path)
	plt.show()

def show_gradebook(gradebook, args, dataset):
	for idx in range(len(dataset)):
	# idx = 1
		sample_data = gradebook.iloc[idx]

		if not (sample_data['accu_hotpot'] == 0.0 and sample_data['accu_tap'] > 0.6):
			continue

		print("idx: ", idx, "/", len(dataset))
		question_id = sample_data['question_id']
		# print(sample_data)
		img_class = dataset[dataset['question_id']==question_id]['image_classes'].item()
		img_id = dataset[dataset['question_id']==question_id]['image_id'].item()
		set_name = dataset[dataset['question_id']==question_id]['set_name'].item()
		if (set_name=="test"):
			set_path = "test/"
			imgset_path = "test_images/"
		else:
			set_path = "train/"
			imgset_path = "train_images/"
		info_path = args.DATAINFO_PATH + set_path + img_id + "_info.npy"
		img_path = args.IMAGE_PATH + imgset_path + img_id + ".jpg"


		# downloaded = 0
		# if os.path.exists(info_path):
		# 	info = np.load(info_path, encoding = "latin1", allow_pickle=True).item()
		# 	downloaded += 1
		visualize_w_pred(sample_data, img_path, info_path, args.GRADEBOOK)




def acc_dist(gradebook):
	accuracy = [0.0, 0.3, 0.6, 0.9, 1.0]
	models = ['hotpot-lorra',
				'hotpot-without-mmt',
				'hotpot-without-object-label',
				'hotpot',
				'object', 'ocr', 'question', # uni-modal models
				'lorra', 'm4c', 'tap'] # competitive models
	# print(gradebook[gradebook['accuracy']==0.0])
	pass

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--GRADEBOOK', '-g',
						help='gradebook name',
						required=True)
	parser.add_argument('--DATAINFO_PATH', '-info',
						help='Data info path',
						default="../../data/")
	parser.add_argument('--IMAGE_PATH', '-i',
						help='Data info path',
						default="../../data/textvqa/")
	parser.add_argument('--SUBSETS', '-s',
						default="obj",
						help='subset name')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	main(args)