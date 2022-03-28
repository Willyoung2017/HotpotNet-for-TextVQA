export CUDA_VISIBLE_DEVICES=${1}
export PYTHONPATH=.
export MMF_CACHE_DIR=./cache/

MODEL=${2}
SAVEDIR=${3}

mmf_run dataset=textvqa_hotpot \
  model=hotpot_net \
  config=${CONFIG} \
  env.save_dir=${SAVEDIR}