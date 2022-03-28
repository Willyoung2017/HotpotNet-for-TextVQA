# script to reproduce m4c and lorra on TextVQA

#----- m4c -----

# m4c evaluate val
mmf_run config=projects/m4c/configs/textvqa/defaults.yaml \
  dataset=textvqa model=m4c env.save_dir=./save/m4c \
  run_type=val checkpoint.resume_zoo=m4c.textvqa.alone

# m4c predict val
mmf_predict config=projects/m4c/configs/textvqa/defaults.yaml \
  datasets=textvqa model=m4c env.save_dir=./save/m4c run_type=val \
  checkpoint.resume_zoo=m4c.textvqa.alone

#----- lorra -----

# download lorra
mkdir -p save/lorra
wget -P -nc save/lorra https://dl.fbaipublicfiles.com/pythia/pretrained_models/textvqa/lorra_best.pth

# lorra evaluate val (w/ vqa_accuracy)
mmf_run config=projects/lorra/configs/textvqa/defaults.yaml \
  dataset=textvqa model=lorra env.save_dir=./save/lorra \
  run_type=val checkpoint.resume_file=./save/lorra/lorra_best.pth

# lorra evaluate val (w/ textvqa_accuracy)
mmf_run config=projects/lorra/configs/textvqa/fixed_eval.yaml \
  dataset=textvqa model=lorra env.save_dir=./save/lorra \
  run_type=val checkpoint.resume_file=./save/lorra/lorra_best.pth

# lorra predict val
mmf_predict config=projects/lorra/configs/textvqa/fixed_eval.yaml \
  dataset=textvqa model=lorra env.save_dir=./save/lorra \
  run_type=val checkpoint.resume_file=./save/lorra/lorra_best.pth
