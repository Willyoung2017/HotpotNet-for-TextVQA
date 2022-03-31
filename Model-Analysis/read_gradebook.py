import pickle
import pandas as pd

df = pd.read_csv('gradebooks/grade_book.csv')
# print(df)

for subset_name in ["position", "obj_related_question"]:
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

	subset_df.to_csv("gradebooks/"+subset_name+"_grade_book.csv")

# read model_preds
with open("model_preds.pickle", "rb") as f:
	model_preds = pickle.load(f)
# print(model_preds)