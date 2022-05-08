'''
This file serves two tasks:
1. gather all model predictions and generate gradebook(s), accuracies and accuracy distribution in whole dataset and subsets
2. visualize questions on challenging questions and worst performing questions

'''

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import glob
import argparse
import re
from tqdm import tqdm


from evaluator import EvalAIAnswerProcessor, TextVQAAccuracyEvaluator

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_pred_path", "-p", default="./model_predictions", help="path of model prediction(s)")
    parser.add_argument("--subset_path", "-s", default="./final_subsets", help="path of file specifying question ids in a subset")
    parser.add_argument("--datainfo_path", "-info", default="../Data/textvqa", help="path of textvqa info data")
    parser.add_argument("--image_path", "-img", default="../../images", help="path of textvqa images")
    parser.add_argument("--task", 
                        default="update", 
                        help="update or viz. update: update grade_book_whole and accuracy_distribution on all subsets, viz: visualize questions on challenging questions and worst performing questions")
    parser.add_argument("--viz_data", "-vd", default="worst", help="worst or hard")
    parser.add_argument("--viz_worst_model", "-vwm", default="hotpot", help="specify worst performing model for visualization")
    parser.add_argument("--viz_question_id", "-v_qid", default=-1, help="the specific question id for visualization")
    parser.add_argument("--update_leaveout_set", "-l", default="leaveout", help="the particular folder under which models are not considered for challenging or worst performing cases")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    eval = TextVQAAccuracyEvaluator() # accuracy evaluator

    val_data = pd.read_json("{}/TextVQA_0.5.1_val.json".format(args.datainfo_path))
    val_gt = val_data['data'].apply(pd.Series) # extract the data portion
    reference = val_gt[["question_id", "image_id", "question", "answers", "question_tokens"]] # extract question id, question text, answers, question_tokens
    question_list = set(reference["question_id"])


    if args.task == "update":
        '''
        update 1. grade_book_whole, 2. distribution, 3. accuracies

        1. grade_book_whole: contains accuracy scores of all models on the entire validation dataset
                columns: 'question_id', 'question', 'answers', 'question_tokens', 'accu_<model name>'

        2. distribution: accuracy distribution of all models on the entire validation dataset, as well as any subset under ./subsets/
        '''
        
        ###############################################
        # 1. Analysis for the Entire Validation Dataset
        ###############################################
        model_pred_paths = glob.glob("{}/*/*.json".format(args.model_pred_path))

        # (1) register models
        models = {}
        for o in model_pred_paths:
            s = o.split("/")
            model_group = s[-2]
            model_name = s[-1].split(".")[0]
            if model_group not in models:
                models[model_group] = [model_name]
            else:
                models[model_group].append(model_name)
        

        # (2) load in prediction results from models
        pred = {}
        for k,v in models.items():
            pred[k] = {i:pd.read_json("./model_predictions/{}/{}.json".format(k,i)) for i in v}
        
        
        # (3) compute accuracy of each model on each question 
        for k,v in pred.items():
            for i in v.keys():

                # sanity check to ensure question ids are the same
                if set(pred[k][i]["question_id"]) != question_list: 
                    # if the list of questions don't match
                    print("*"*20)
                    print("WARNING! The number of questions in the output of {}/{} doesn't match with validation dataset".format(k, i))
                    print("*"*20)
                    
                pred[k][i] = pred[k][i].merge(reference, on='question_id', how='inner')
                pred[k][i].rename(columns={"answer":"prediction"}, inplace=True)
                pred[k][i]["accu_"+i] = pred[k][i].apply(lambda row: round(eval.eval_pred_list([{"pred_answer":row.prediction, "gt_answers":row.answers}]), 1), axis=1)
                

        # (4) generate gradebook of all models on all questions
        gradebook = reference.copy()
        gradebook_with_pred = reference.copy() 
        gradebook_no_leaveout_set = reference.copy()
        gradebook_with_pred_no_leaveout_set = reference.copy() 

        for k,v in tqdm(pred.items()):
            for i in v.keys():
                temp = v[i].rename(columns={"prediction":"pred_"+i, "pred_source":"pred_source_"+i})
                print("{}   {}: {}".format(k, i, np.mean(pred[k][i]["accu_"+i])))
                gradebook = gradebook.merge(right=pred[k][i][["question_id", "accu_"+i]], on="question_id", how="inner")
                gradebook_with_pred = gradebook_with_pred.merge(right=pred[k][i][["question_id", "accu_"+i]], on="question_id", how="inner")
                gradebook_with_pred = gradebook_with_pred.merge(temp[["question_id", "pred_"+i, "pred_source_"+i]], on="question_id", how="inner")

                if k != args.update_leaveout_set:
                    gradebook_no_leaveout_set = gradebook_no_leaveout_set.merge(right=pred[k][i][["question_id", "accu_"+i]], on="question_id", how="inner")
                    gradebook_with_pred_no_leaveout_set = gradebook_with_pred_no_leaveout_set.merge(right=pred[k][i][["question_id", "accu_"+i]], on="question_id", how="inner")
                    gradebook_with_pred_no_leaveout_set = gradebook_with_pred_no_leaveout_set.merge(temp[["question_id", "pred_"+i, "pred_source_"+i]], on="question_id", how="inner")

        gradebook.to_csv("./final_gradebooks/gradebook_whole.csv", index=False)
        gradebook_with_pred.to_csv("./final_gradebooks/gradebook_with_pred_whole.csv", index=False)

        accuracies_no_leaveout_set = gradebook_no_leaveout_set.iloc[:, 5:]

        # (5) compute accuracy
        accuracies = gradebook.iloc[:, 5:]
        accu = accuracies.mean().to_frame().T
        accu.insert(0, "dataset", "whole")
        accu_result = [accu] # contains model accuracies on whole validation dataset, and various subsets


        # (6) compute and save accuracy distribution 
        accu_dist = accuracies.apply(pd.Series.value_counts).sort_index(ascending=False)
        accu_dist.index.rename("accuracy", inplace=True)
        accu_dist.to_csv("./final_accuracies/accu_dist_whole.csv")


        # (7) generate gradebooks with predictions for all-zero questions, and worst performing questions
        if args.update_leaveout_set == None:
            accuracies_no_leaveout_set = accuracies
            gradebook_with_pred_no_leaveout_set = gradebook_with_pred
            gradebook_no_leaveout_set = gradebook

        max = accuracies_no_leaveout_set.max(axis=1)
        min = accuracies_no_leaveout_set.min(axis=1)
        sec_min = accuracies_no_leaveout_set.apply(lambda row: row.nsmallest(2).values[-1], axis=1)

        all_zero_gradebook_with_pred = gradebook_with_pred_no_leaveout_set[max+min==0]

        unique_zero_gradebook = gradebook_no_leaveout_set[(min==0) & (sec_min>0)]
        unique_zero_gradebook_with_pred = gradebook_with_pred_no_leaveout_set[(min==0) & (sec_min>0)]
        unique_zero_gradebook_with_pred["worst_model"] = unique_zero_gradebook.iloc[:,5:].idxmin(axis="columns")
        unique_zero_gradebook_with_pred["worst_model"] = unique_zero_gradebook_with_pred["worst_model"].apply(lambda x: x[5:])
        worst_count = pd.value_counts(unique_zero_gradebook_with_pred["worst_model"]).to_frame().T

        all_zero_gradebook_with_pred.to_csv("./final_gradebooks/gradebook_with_pred_challenging.csv", index=False)
        unique_zero_gradebook_with_pred.to_csv("./final_gradebooks/gradebook_with_pred_worst_performing.csv", index=False)
        worst_count.to_csv("./final_accuracies/worst_count.csv", index=False)


        ###############################################
        # 2. Analysis for Subsets
        ###############################################
        subset_paths = glob.glob("{}/*.txt".format(args.subset_path))

        for s in subset_paths:
            subset = re.split('\W', s)[-2]
            f = open(s, "r")
            sub_q = f.read().split(",")  
            sub_gradebook = gradebook[gradebook["question_id"].isin([int(d) for d in sub_q])]
            sub_gradebook.to_csv("./final_gradebooks/gradebook_{}.csv".format(subset), index=False)

            # compute accuracy
            sub_accuracies = sub_gradebook.iloc[:, 5:]
            sub_accu = sub_accuracies.mean().to_frame().T
            sub_accu.insert(0, "dataset", subset)
            accu_result.append(sub_accu)
            
            # compute and save accuracy distribution 
            sub_accu_dist = sub_accuracies.apply(pd.Series.value_counts).sort_index(ascending=False)
            sub_accu_dist.index.rename("accuracy", inplace=True)
            sub_accu_dist.to_csv("./final_accuracies/accu_dist_{}.csv".format(subset))
        

        # save accuracy result
        accu_result = pd.concat(accu_result, ignore_index=True)
        accu_result.to_csv("./final_accuracies/accuracies.csv", index=False)



    elif args.task == "viz":
        if args.viz_question_id != -1:
            gradebook_with_pred = pd.read_csv("./final_gradebooks/gradebook_with_pred_whole.csv")
            gb = gradebook_with_pred[gradebook_with_pred["question_id"] == int(args.viz_question_id)]
            for i in range(gb.shape[0]):
                data = gb.iloc[i,:]
                image_id = data["image_id"]
                img = plt.imread("../../images/train_images/{}.jpg".format(image_id))

                # print question, answer, predictions and accuracy scores
                pd.set_option('display.max_colwidth', None)
                print ("progress = {}/{}".format(i, gb.shape[0]))
                print (data)
                
                fig = plt.imshow(img)
                plt.axis("off")
                plt.show()

                print("\n")
                print ("*"*100)
                print ("*"*100)
                print("\n")
        else:
            if args.viz_data == "hard":
                gradebook_with_pred_challenging = pd.read_csv("./final_gradebooks/gradebook_with_pred_challenging.csv")
                for i in range(gradebook_with_pred_challenging.shape[0]):
                    data = gradebook_with_pred_challenging.iloc[i,:]
                    image_id = data["image_id"]
                    img = plt.imread("../../images/train_images/{}.jpg".format(image_id))

                    # print question, answer, predictions and accuracy scores
                    pd.set_option('display.max_colwidth', None)
                    print ("progress = {}/{}".format(i, gradebook_with_pred_challenging.shape[0]))
                    print (data)
                    
                    fig = plt.imshow(img)
                    plt.axis("off")
                    plt.show()

                    print("\n")
                    print ("*"*100)
                    print ("*"*100)
                    print("\n")
            
            else:
                gradebook_with_pred_worst = pd.read_csv("./final_gradebooks/gradebook_with_pred_worst_performing.csv")
                gb_worst_model = gradebook_with_pred_worst[gradebook_with_pred_worst["worst_model"]==args.viz_worst_model]
                for i in range(gb_worst_model.shape[0]):
                    data = gb_worst_model.iloc[i,:]
                    image_id = data["image_id"]
                    img = plt.imread("../../images/train_images/{}.jpg".format(image_id))

                    # print question, answer, predictions and accuracy scores
                    pd.set_option('display.max_colwidth', None)
                    print ("progress = {}/{}".format(i, gb_worst_model.shape[0]))
                    print (data)
                    
                    fig = plt.imshow(img)
                    plt.axis("off")
                    plt.show()

                    print("\n")
                    print ("*"*100)
                    print ("*"*100)
                    print("\n")

    else:
        pass


if __name__ == '__main__':
    main()