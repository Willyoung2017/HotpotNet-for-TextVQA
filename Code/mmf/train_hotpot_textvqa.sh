export CUDA_VISIBLE_DEVICES=${1}
export PYTHONPATH=.
export MMF_CACHE_DIR=./cache/

MODEL=${2}
CONFIG=${3}
SAVEDIR=${4}

mmf_run dataset=textvqa_hotpot \
  model=${MODEL} \
  config=${CONFIG} \
  env.save_dir=${SAVEDIR}
