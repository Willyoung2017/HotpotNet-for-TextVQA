This folder contains code and results of qualitative and quantitative model analysis. 

## File Structure
- `model_predictions`: finalized prediction results of unimodal baselines, competitive baselines, hotpot baselines and finalized hotpot models
- Quantitative Analysis Results:
    - `final_gradebooks`: finalized gradebooks containing accuracy scores of all models on whole validation dataset as well as subsets
    - `final_accuracies`: finalized accuracies and accuracy distributions of all models
    - `final_subsets`: finalized subsets of validation dataset that are of interest in this research project. 
- `intermediate_code_and_results`: (outdated) intermediate code and results of subsets, baseline model predictions and gradebooks
- Code for Analysis:
    - `dataset_splitter.py`: generate subset
    - `analysis.py`: 
        -  generate gradebook, summary figures (accuracies), accuracy distributions
        -  visualize image and print question data (question text, ground truth answer, model predictions, model accuracy scores)

## Pipeline and File Usage
- Make sure your working directory is under `Model-Anlaysis/`
- Put model prediction results under `model_predictions`.
- To update quantitative results (gradebook, accuracies and accuracy distribution), execute the following command:
    
    `python analysis.py --task update`
    
    if you want to exclude certain models from being considered in worst performing questions and challenging questions (but still want their accuracies), execute the following command:
    
    `python analysis.py --task update --update_leaveout_set <folder name under ./model_predictions>`

- To visualize questions and select failure cases, execute the following commands:
    - To visualize one particular question, you need to specify the question id (e.g. 37886):  

    `python analysis.py --task viz --viz_question_id 37886`
    
    - To sequentially visualize all challenging questions:  

    `python analysis.py --task viz --viz_data hard`
    
    - To sequentially visualize worst performing questions for a specific model, you need to specify the name of model as denoted in `.json` file (e.g. "hotpot-lorra"):  

    `python analysis.py --task viz --viz_data worst --viz_worst_model hotpot-lorra`
    
    
---


If using notebooks, please commit them here with plots/graphs/statistics annotated with inline comments.  

If you instead use command-line tools, please describe how to run your code, and upload generated plots and statistics with documentation explaining them.



All these utils are applied to `val` dataset right now.



`dataset_splitter.py` generates subset where there are certain words in questions.
- Split **OCR** subset: `python dataset_splitter.py -o`
- Split subset based on keyword in **question** or **answer** field
  - Generate your own *keywordlist.txt*, put it in the `subset_filters` folders.
  - `python dataset_splitter.py -k [your_keyword_list_name]`, your keywrod list is your filter, which is used to filter the dataset. 
  - The output will be two `.txt` files
    - One contains all the `question_id` that meets your requirement.
    - The other one show the `question` with the `question_id`



`datainfo_collector.py`  collets data information from `_info.npy` file

* INFO_DIRECTORY should include two subfolders `train/` and `test/`, where you put the `_info.npy` files
* `python datainfor_collector.py -info [path_to_INFO_DIRECTORY]`
* The output will be two `.txt` files
  * One contains all the `question_id` that meets your requirement.
  * The other one show the `question` with the `question_id`



`read_gradebook.py` generates gradebook for your subset

* One argument `-s [your_subset_names]` is required. You can generate several gradebook for subsets, you should use exactly subset name from `subsets` folder and split them with comma, for example, for generating gradebooks for subset `not require reading` and `require reading`, you put in argument `-s "not_require_reading,require_reading"`.
* The output will be `.csv` files, placed under `gradebooks` folder.
