#!/bin/bash


SAVE_DIR="/home/rita/cantonese-asr/checkpoints/}"

DATA_ROOT="/home/rita/cantonese-asr/CoVoHK"


CODE_DIR=/home/rita/cantonese-asr/fairseq/../../
cd ${CODE_DIR}


# for i in ${DATA_ROOT}/clips/*.mp3
# do
#     sox "$i" "${DATA_ROOT}/waves/$(basename -s .mp3 "$i").wav"
# done

# cd fairseq
# pip install --src=/home/rita/cantonese-asr/fairseq --editable ./
# python /home/rita/cantonese-asr/fairseq/setup.py build develop
# PYTHONPATH=/home/rita/cantonese-asr/fairseq python -m fairseq_cli.train


#data preprocessing
python /home/rita/cantonese-asr/fairseq/examples/speech_to_text/prep_covost_canto_asr_data.py \
    --data-root ${DATA_ROOT}   --vocab-type char --vocab-size 4000 --character-coverage 0.9995