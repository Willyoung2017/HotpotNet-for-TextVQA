# script to reproduce TAP on TextVQA
# note: run this script from the root folder of TAP

# 1. run the following commands to create conda env:
#    conda create -n tap python=3.6
#    conda activate tap
#    conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
#    pip install setuptools==58
#    pip install -r requirements.txt

# 2. rewrite pythia/utils/configuration.py to remove dependency on demjson
python -u tools/run.py --tasks vqa --datasets m4c_textvqa \
  --model m4c_split --config configs/vqa/m4c_textvqa/tap_refine.yml \
  --save_dir save/m4c_split_refine_test --run_type val \
  --resume_file save/finetuned/textvqa_tap_base_best.ckpt
