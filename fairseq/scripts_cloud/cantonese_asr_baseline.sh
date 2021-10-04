#!/bin/bash

SAVE_DIR="home/rita/cantonese-asr/checkpoints"

DATA_ROOT="home/rita/cantonese-asr/dataset"

cd fairseq
pip install --editable ./

CUDA_VISIBLE_DEVICES=0
fairseq-train ${DATA_ROOT} --train-subset cnt_asr_train_metadata --valid-subset cnt_asr_valid_metadata \
    --task speech_to_text --config-yaml config_asr.yaml  \
    --num-workers 4 --max-tokens 40000 --max-update 300000 \ 
    --arch s2t_transformer_s --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --report-accuracy --share-decoder-input-output-embed \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
    --max-tokens 12000 --clip-norm 10.0 --seed 1 --update-freq 8 --save-dir ${SAVE_DIR}
    
    
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py --inputs ${SAVE_DIR} \
  --num-epoch-checkpoints 10 \
  --output "${SAVE_DIR}/${CHECKPOINT_FILENAME}"
for SUBSET in dev-clean dev-other test-clean test-other; do
  fairseq-generate ${LS_ROOT} --config-yaml config.yaml --gen-subset ${SUBSET} \
    --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring wer
done
    