import numpy as np
import pandas as pd
import csv
import json



def read_obj_vocab(vocab_path):
	with open(vocab_path, 'r') as f:
		obj_vocab = f.read().splitlines()
	obj_vocab = ["BACKGROUND"] + obj_vocab
	bgindex = obj_vocab.index("background")
	print("object volab len:", len(obj_vocab))
	print("background index:", bgindex)
	return obj_vocab


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
	selected_columns = ['question_id',
						'question',
						'image_classes',
						'image_id',
						'set_name']

	# val_text = val_data[selected_columns]
	# train_text = train_data[selected_columns]
	# test_text = test_data[selected_columns]

	return train_data, val_data, test_data