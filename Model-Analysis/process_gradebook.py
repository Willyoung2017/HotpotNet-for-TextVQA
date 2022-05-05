from ast import arg
import pandas as pd
import numpy as np
import pickle
import argparse
import os
import random
from pathlib import Path
# import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


from utils import read_dataset, read_obj_vocab
from visualization import visualize_w_bb, visualize_img, visualize_w_qa
from evaluator import EvalAIAnswerProcessor, TextVQAAccuracyEvaluator
from read_gradebook import generate_gradebook


def show_gradebook(gradebook, args, dataset):
	for idx in range(len(dataset)):
	# idx = 1
		sample_data = gradebook.iloc[idx]
		print(sample_data)
		# if not (sample_data['accu_hotpot'] == 0.0 and sample_data['accu_tap'] > 0.6):
		# 	continue

		print("idx: ", idx, "/", len(dataset))
		question_id = sample_data['question_id']

		image_classes = dataset[dataset['question_id']==question_id]['image_classes'].item()
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
		
		info = np.load(info_path, encoding = "latin1", allow_pickle=True).item()

		visualize_w_qa(sample_data, img_path, info, args.GRADEBOOK)

def gen_gradebook_w_pred(gradebook, args):
	# print(gradebook.columns)

	with open(args.PREDICTION, "r") as pred_file:
		pred_df = pd.read_json(pred_file)
	pred_file.close()
	# # question_id          image_id                    answer          pred_source
	# print(pred_df)
			
	gb_pred = gradebook[['question_id', 'question', 'answers']].copy() # save the accuracy of all models on all questions
	gb_pred = gb_pred.merge(pred_df, on='question_id', how='inner')
	gb_pred.rename(columns={"answer":"prediction"}, inplace=True)
	
	eval = TextVQAAccuracyEvaluator()
		
	gb_pred["pred_score"] = gb_pred.apply(lambda row: eval.eval_pred_list([{"pred_answer":row.prediction, "gt_answers":row.answers}]), axis=1)
	print(gb_pred)
	print("Subset accuracy: ", gb_pred['pred_score'].sum()/len(gb_pred))
	# gb_pred.to_csv("gradebooks/gb_" + args.GRADEBOOK + "_with_pred.csv", index=False)


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
	parser.add_argument('--PREDICTION', '-p',
						help='prediction file',
						default="./pred/multimodal-baselines/textvqa-val-hotpot.json")
	parser.add_argument('--DATAINFO_PATH', '-info',
						help='Data info path',
						default="../../data/")
	parser.add_argument('--IMAGE_PATH', '-i',
						help='Data info path',
						default="../../data/textvqa/")
	parser.add_argument("--VISUALIZATION", '-v',
						help="Visualize gradebook samples",
						default=False,
						action='store_true')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	dataset_path = "../Data/textvqa/"
	trainset, valset, testset = read_dataset(dataset_path)
	obj_vocab = read_obj_vocab("../Data/objects_vocab.txt")

	# # baseline model zoo
	# models = {}
	# models["multimodal-baselines"] = ["hotpot-lorra", "hotpot-without-mmt", "hotpot-without-object-label", "hotpot"]
	# models["unimodal-baselines"] = ["object", "ocr", "question"]
	# models["competitive-baselines"] = ["lorra", "m4c", "tap"]

	# gradebook_path = "gradebooks/grade_book_" + args.GRADEBOOK + ".csv"
	# gradebook = pd.read_csv(gradebook_path)


	# print(valset)
	gradebook = generate_gradebook(args.GRADEBOOK,valset)

	if args.VISUALIZATION:
		show_gradebook(gradebook, args, valset)

	# print(gradebook)
	gen_gradebook_w_pred(gradebook, args)