This folder is for performing model analysis.

If using notebooks, please commit them here with plots/graphs/statistics annotated with inline comments.  

If you instead use command-line tools, please describe how to run your code, and upload generated plots and statistics with documentation explaining them.



All these utils are applied to `val` dataset right now.



`dataset_splitter.py` generates subset where there are certain words in questions.

- Generate your own *keywordlist.txt*, put it in the same directory with your splitter
- `python dataset_splitter.py -k [your_keyword_list_txt_file]`
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
