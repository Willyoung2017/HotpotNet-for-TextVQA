import pickle
import pandas as pd
import argparse

df = pd.read_csv('gradebooks/grade_book_whole.csv')

parser = argparse.ArgumentParser()

parser.add_argument('--SUBSETS', '-s',
					help='subset names splitted with comma',
					required=True)

args = parser.parse_args()
subset_list = args.SUBSETS.split(",")
print(subset_list)

for subset_name in subset_list:
	subset_path = "subsets/val_" + subset_name + ".txt"
	with open(subset_path, "r") as f:
		subset = f.read()
	subset = subset.split(",")
	subset = [int(x) for x in subset]
	# print(subset)

	subset_df = df[df["question_id"].isin(subset)]
	# print(subset_df.columns)
	# print(subset_df[['accu_hotpot-lorra', 'accu_hotpot-without-mmt',
	# 	'accu_hotpot-without-object-label', 'accu_hotpot', 'accu_object',
	# 	'accu_ocr', 'accu_question', 'accu_lorra', 'accu_m4c', 'accu_tap']])

	subset_df.to_csv("gradebooks/grade_book_"+subset_name+".csv")

# read model_preds
# with open("model_preds.pickle", "rb") as f:
# 	model_preds = pickle.load(f)
# print(model_preds)