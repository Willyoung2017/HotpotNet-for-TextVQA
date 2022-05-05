import pickle
import pandas as pd
import argparse


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--SUBSETS', '-s',
						help='subset names splitted with comma',
						required=True)
	args = parser.parse_args()
	subset_list = args.SUBSETS.split(",")
	print(subset_list)

	df = pd.read_csv('gradebooks/grade_book_whole.csv')
	subset_statistics = dict()
	for subset_name in subset_list:
		subset_df = generate_gradebook(subset_name, df)
		subset_accu = analyze_gradebook(subset_df)
		subset_statistics[subset_name] = subset_accu
	
	subset_performance_df = pd.DataFrame(subset_statistics)
	print(subset_performance_df)

def generate_gradebook(subset_name, df):
	subset_path = "subsets/val_" + subset_name + ".txt"
	with open(subset_path, "r") as f:
		subset = f.read()
	f.close()
	
	subset = subset.split(",")
	subset = [int(x) for x in subset]

	subset_df = df[df["question_id"].isin(subset)]
	# print(subset_df.columns)
	# print(subset_df[['accu_hotpot-lorra', 'accu_hotpot-without-mmt',
	# 	'accu_hotpot-without-object-label', 'accu_hotpot', 'accu_object',
	# 	'accu_ocr', 'accu_question', 'accu_lorra', 'accu_m4c', 'accu_tap']])

	subset_df.to_csv("gradebooks/grade_book_"+subset_name+".csv")
	return subset_df

def analyze_gradebook(subset_df):
	performance_dict = dict()
	model_list = ['accu_hotpot-lorra', 'accu_hotpot-without-mmt',
		'accu_hotpot-without-object-label', 'accu_hotpot', 'accu_object',
		'accu_ocr', 'accu_question', 'accu_lorra', 'accu_m4c', 'accu_tap']
	for m in model_list:
		performance_dict[m] = subset_df[m].sum()/len(subset_df)
	
	return performance_dict

if __name__ == '__main__':
	main()