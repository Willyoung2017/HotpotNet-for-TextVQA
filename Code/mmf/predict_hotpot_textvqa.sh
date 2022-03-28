export CUDA_VISIBLE_DEVICES=${1}
export PYTHONPATH=.
export MMF_CACHE_DIR=./cache/

MODEL=${2}
SAVEDIR=${3}

mmf_predict dataset=textvqa_hotpot \
  model=hotpot_net \
  config=${CONFIG} \
  env.save_dir=${SAVEDIR} \
  run_type=val \
  checkpoint.resume=True \
  checkpoint.resume_best=True