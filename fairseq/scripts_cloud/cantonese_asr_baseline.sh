#!/bin/bash

model_name=covost_asr_transformer_xs_lr_0.003_char_no_specaugment
# rm -R "/home/rita/cantonese-asr/checkpoints/${model_name}"
mkdir "/home/rita/cantonese-asr/checkpoints/${model_name}"
SAVE_DIR="/home/rita/cantonese-asr/checkpoints/${model_name}"

DATA_ROOT="/home/rita/cantonese-asr/dataset"
mkdir "/home/rita/cantonese-asr/checkpoints/${model_name}/tensorboard/"
TENSORBOARD_LOGDIR="/home/rita/cantonese-asr/checkpoints/${model_name}/tensorboard/"

# cd fairseq
# pip install --src=/home/rita/cantonese-asr/fairseq --editable ./
# python /home/rita/cantonese-asr/fairseq/setup.py build develop
# PYTHONPATH=/home/rita/cantonese-asr/fairseq python -m fairseq_cli.train
CODE_DIR=/home/rita/cantonese-asr/fairseq/../../
cd ${CODE_DIR}

# #data preprocessing
# python /home/rita/cantonese-asr/fairseq/examples/speech_to_text/prep_cantonese_asr_data.py \
#     --data-root ${DATA_ROOT}   --vocab-type unigram --vocab-size 8000 --character-coverage 0.9995


#training
export CUDA_VISIBLE_DEVICES=4,5,6,7
python /home/rita/cantonese-asr/fairseq/fairseq_cli/train.py ${DATA_ROOT} --train-subset train_asr --valid-subset dev_asr \
    --task speech_to_text --config-yaml config_asr_chars.yaml --tensorboard-logdir  ${TENSORBOARD_LOGDIR} --keep-last-epochs 10 \
    --label-smoothing 0.1 --report-accuracy --share-decoder-input-output-embed   --batch-size 16 \
    --optimizer adam --lr 0.003 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
     --criterion label_smoothed_cross_entropy \
     --arch s2t_transformer_xs   --num-workers 1 --patience 10 \
    --clip-norm 10.0 --seed 1 --update-freq  2 --save-dir ${SAVE_DIR}
    

    
# # #checkpoint averaging    
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python /home/rita/cantonese-asr/fairseq/scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 10 \
  --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
  
#testing      
python /home/rita/cantonese-asr/fairseq/fairseq_cli/generate.py ${DATA_ROOT} --config-yaml config_asr_chars.yaml --gen-subset test_asr \
    --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --arch s2t_transformer_xs  \
   --batch-size 16 --beam 5 --scoring wer  --wer-char-level --results-path $SAVE_DIR 

# --wer-char-level
generated_file=${SAVE_DIR}/generate-test_asr.txt

cat ${generated_file} | awk 'END{print}'