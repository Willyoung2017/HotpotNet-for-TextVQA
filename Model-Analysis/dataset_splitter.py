import numpy as np
import json
import csv
import pandas as pd
import argparse
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
	args = parser.parse_args()

	trainset, valset, testset = read_dataset(dataset_path)

	subsetname = args.KEYWORD_LIST
	keywordlist = read_keywords(args)
	print("keyword: ", keywordlist)

	if (args.ANSWER_FIELD):
		print("Keyword searching in Answer texts ...")
		subset = split_dataset_answer(valset, subsetname, keywordlist)
	else:
		print("Keyword searching in Question texts ...")
		subset = split_dataset_question(valset, subsetname, keywordlist)
	
	save_subset("val", subsetname, subset)


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

	val_text = val_data[['question_id', 'question', 'answers']]
	train_text = train_data[['question_id', 'question', 'answers']]
	test_text = test_data[['question_id', 'question']]

	return train_text, val_text, test_text


if __name__ == '__main__':
	main()



