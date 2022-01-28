#!/bin/bash

model_name=joint_asr_transformer_xs_lr_0.003_vocab_8000_no_specaugment
# rm -R "/home/rita/cantonese-asr/checkpoints/${model_name}"
mkdir "/home/rita/cantonese-asr/checkpoints/${model_name}"
SAVE_DIR="/home/rita/cantonese-asr/checkpoints/${model_name}"

# DATA_ROOT="/home/rita/cantonese-asr/dataset"
DATA_ROOT="/home/rita/cantonese-asr/Joint_data"
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
export CUDA_VISIBLE_DEVICES=1
# python /home/rita/cantonese-asr/fairseq/fairseq_cli/train.py ${DATA_ROOT} --train-subset train_combined --valid-subset dev_combined \
#     --task speech_to_text --config-yaml config_asr_no_specaugment.yaml --tensorboard-logdir  ${TENSORBOARD_LOGDIR} --keep-last-epochs 10 \
#     --label-smoothing 0.1 --report-accuracy --share-decoder-input-output-embed   --batch-size 64 \
#     --optimizer adam --lr 0.003 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
#      --criterion label_smoothed_cross_entropy \
#      --arch s2t_transformer_xs   --num-workers 1 --patience 10 \
#     --clip-norm 10.0 --seed 1 --update-freq  4 --save-dir ${SAVE_DIR}
    

    
# # #checkpoint averaging    
# export CUDA_VISIBLE_DEVICES=6 
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python /home/rita/cantonese-asr/fairseq/scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 10 \
  --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
  
#testing  

for test in test_new_asr ;
do
python /home/rita/cantonese-asr/fairseq/fairseq_cli/generate.py ${DATA_ROOT} --config-yaml config_asr_no_specaugment.yaml --gen-subset ${test} \
    --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --arch s2t_transformer_xs  \
   --batch-size 64 --beam 8 --scoring wer --wer-char-level --results-path $SAVE_DIR 
done

# --wer-char-level
# rm ${SAVE_DIR}/generate-test-asr-CoVoHK.txt
generated_file=${SAVE_DIR}/generate-test_ours.txt
generated_file_CoVoHK=${SAVE_DIR}/generate-test_new_asr.txt
generated_joint_file=${SAVE_DIR}/generate-test_combined.txt 
echo  "Ours"
cat ${generated_file} | awk 'END{print}'
echo "Common Voice HK"
cat ${generated_file_CoVoHK} | awk 'END{print}'
echo "Joint"
cat ${generated_joint_file} | awk 'END{print}'