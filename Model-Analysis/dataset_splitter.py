import numpy as np
import json
import csv
import pandas as pd
import argparse
from evaluator import EvalAIAnswerProcessor
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sb
# import spacy


def main():
	dataset_path = "../Data/textvqa/"
	parser = argparse.ArgumentParser()

	parser.add_argument('--KEYWORD_LIST', '-k', help='Keywords list file')
	parser.add_argument("--QUESTION_FIELD", '-q',
							help="Search for questions that include keyword",
							default=False,
							action='store_true')
	parser.add_argument("--ANSWER_FIELD", '-a',
							help="Search for questions whose answers include keyword",
							default=False,
							action='store_true')
	parser.add_argument("--OCR_FIELD", '-o',
							help="Split for OCR subset",
							default=False,
							action='store_true')
	args = parser.parse_args()

	trainset, valset, testset = read_dataset(dataset_path)

	if (args.OCR_FIELD):
		print("Splitting subset based on OCR tokens ...")
		rosetta_path = dataset_path + "TextVQA_Rosetta_OCR_v0.2_val.json"
		msocr_path = "../Data/textvqa_msocr/train_images/"
		ocr_dict = split_dataset_ocr(valset, rosetta_path, msocr_path)
		for subsetname, subset in ocr_dict.items():
			print(subsetname)
			save_subset("val", subsetname, subset)
	else:
		subsetname = args.KEYWORD_LIST
		keywordlist = read_keywords(args)
		print("keyword: ", keywordlist)

		if (args.ANSWER_FIELD):
			print("Keyword searching in Answer texts ...")
			subset = split_dataset_answer(valset, subsetname, keywordlist)
		elif (args.QUESTION_FIELD):
			print("Keyword searching in Question texts ...")
			subset = split_dataset_question(valset, subsetname, keywordlist)
	
		save_subset("val", subsetname, subset)

def split_dataset_ocr(dataset, rosetta_path, msocr_path):
	no_contain_gt_index_list = []
	both_contain_gt_index_list = []
	rosetta_contain_gt_index_list = []
	msocr_contain_gt_index_list = []
	with open(rosetta_path) as rosetta_file:
		rosetta_data_list = json.load(rosetta_file)
	rosetta_file.close()
	rosetta_df = pd.json_normalize(rosetta_data_list, record_path='data')
	# # Index(['question', 'image_id', 'image_classes', 'flickr_original_url',
    # #    'flickr_300k_url', 'image_width', 'image_height', 'answers',
    # #    'question_tokens', 'question_id', 'set_name'],
    # #   dtype='object')
	# print(rosetta_df)

	has_no_rosetta = []
	for idx in range(len(dataset)):
		image_id = dataset.iloc[idx]['image_id']
		answers = dataset.iloc[idx]['answers']

		answer_processor = EvalAIAnswerProcessor()
		msocr_info = np.load(msocr_path + str(image_id) + "_info.npy", encoding = "latin1", allow_pickle=True).item()
		msocr_answers = [answer_processor(x) for x in msocr_info['ocr_tokens']]

		both_not_contain = True
		for a in msocr_answers:
			if (a in answers):
				both_not_contain = False
				msocr_contain_gt_index_list.append(idx)
				continue
		
		rosetta_info = rosetta_df[rosetta_df['image_id'] == image_id].iloc[0]
		rosetta_answers = [answer_processor(x) for x in rosetta_info['ocr_tokens']]
		if (len(rosetta_answers) == 0):
			# print(image_id, " has no rosetta OCR data")
			has_no_rosetta.append(image_id)
			continue
		for a in rosetta_answers:
			if (a in answers):
				both_not_contain = False
				rosetta_contain_gt_index_list.append(idx)
				continue
		
		if both_not_contain:
			no_contain_gt_index_list.append(idx)
		
	print(len(has_no_rosetta), "images has no rosetta data")
	both_contain_gt_index_list = list(set(rosetta_contain_gt_index_list) & set(msocr_contain_gt_index_list))

	# Differnt OCR
	for x in both_contain_gt_index_list:
		rosetta_contain_gt_index_list.remove(x)
		msocr_contain_gt_index_list.remove(x)
			
	print(len(no_contain_gt_index_list), "samples, no OCR system contains any ground truth")
	print(len(both_contain_gt_index_list), "samples, both OCR system contains ground truth")
	print(len(rosetta_contain_gt_index_list), "samples, only Rosetta OCR system contains ground truth")
	print(len(msocr_contain_gt_index_list), "samples, only MS OCR system contains ground truth")

	ocr_subset_dict = dict()
	ocr_subset_dict['no_ocr_contain_gt'] = dataset.iloc[no_contain_gt_index_list]
	ocr_subset_dict['both_ocr_contain_gt'] = dataset.iloc[both_contain_gt_index_list]
	ocr_subset_dict['rosetta_contain_gt'] = dataset.iloc[rosetta_contain_gt_index_list]
	ocr_subset_dict['msocr_contain_gt'] = dataset.iloc[msocr_contain_gt_index_list]
	# print(ocr_subset_dict)

	return ocr_subset_dict
		

def split_dataset_answer(dataset, subsetname, keywordlist):
	# for idx in range(len(dataset)):
	index_list = []
	for idx in range(len(dataset)):
		# ans_str = "".join([x+" " for x in dataset.iloc[idx]['answers']])
		# for split_filter in keywordlist:
		# 	if (split_filter in ans_str):
		# 		index_list.append(idx)
		# 		continue

		answers = dataset.iloc[idx]['answers']
		for split_filter in keywordlist:
			if (split_filter in answers):
				index_list.append(idx)
				continue
	subset = dataset.iloc[index_list]
	print(subset)
	return subset

def split_dataset_question(dataset, subsetname, keywordlist):
	split_filter = "".join([keyword + "|" for keyword in keywordlist])
	split_filter = split_filter[:-1]
	subset = dataset[dataset['question'].str.contains(split_filter, regex=True)]
	# print(subsetname, ":", len(dataset), "->", len(subset))
	print(subset)
	return subset

def save_subset(subset_source, subsetname, subset):
	subset_str = "".join([str(x)+"," for x in set(subset['question_id'])])
	subset_str = subset_str[:-1]
	with open("subsets/"+subset_source+"_"+subsetname+".txt", 'w') as f:
		f.write(subset_str)
	f.close()

	subset_str_text = ""
	for i in range(len(subset)):
		subset_str_text += str(subset.iloc[i]['question_id']) + "\t" + str(subset.iloc[i]['question'] +"\n")
	with open("subsets/"+subset_source+"_"+subsetname+"_text.txt", 'w') as f:
		f.write(subset_str_text)
	f.close()

def read_keywords(args):
	keyword_list_path = "subset_filters/" + args.KEYWORD_LIST + ".txt"
	with open(keyword_list_path, 'r') as f:
		keylist = f.read().splitlines()
	f.close()
	return keylist

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

	# Index(['question', 'image_id', 'image_classes', 'flickr_original_url',
	# 	'flickr_300k_url', 'image_width', 'image_height', 'answers',
	# 	'question_tokens', 'question_id', 'set_name'],
	# 	dtype='object')

	val_text = val_data[['image_id', 'image_classes', 'question_id', 'question', 'answers', 'set_name']]
	train_text = train_data[['image_id', 'image_classes', 'question_id', 'question', 'answers', 'set_name']]
	test_text = test_data[['image_id', 'image_classes', 'question_id', 'question', 'set_name']]

	return train_text, val_text, test_text


if __name__ == '__main__':
	main()



