CUDA_VISIBLE_DEVICES=1 /online/speech/peihao.wu/anaconda2/bin/python generator_covert_model.py \
  --type=eval \
  --config=../conf/single.conf \
  --evalfile=$1
