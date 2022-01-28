#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from tqdm import tqdm
import numpy as np
import pandas as pd
import soundfile as sf
import os
from itertools import groupby
from typing import Tuple
import glob

from data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)

import torch
from torch.utils.data import Dataset






log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]


class Cantonese_ASR(Dataset):
    """
    Create a Dataset for Cantonese_ASR. Each item is a tuple of the form:
    waveform, sample_rate, target utterance,
    utterance_id
    """
    SPLITS = ["train"]



    def __init__(self, root: str,split: str, args) -> None:
        assert split in self.SPLITS 
        _root = Path(root) 
        wav_root = _root / "waves"
        # text_files=glob.glob('**/*.txt')
        # wav_files=glob.glob('**/*.wav')
        assert _root.is_dir() 
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files")
        # with open(txt_root / f"{split}.yaml") as f:
        
        tsv_file= _root/ f"train_combined.tsv"
        segments = pd.read_csv(tsv_file, sep='\t')
        
       
        #generate sentencepiece vocabulary from the segments dataframe target text column
        #generate sentencepiece model from the segments dataframe target text column

        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        #taking argparse argument vocab_type and vocab_size to generate the vocab file name
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_joint_asr"

        with NamedTemporaryFile(mode="w") as f:
            for t in segments['tgt_text']:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                _root / spm_filename_prefix,
                args.vocab_type, 
                args.vocab_size,
                args.character_coverage,
                
            )
    # Generate config YAML
    # if args.use_audio_input:
    #     gen_config_yaml(
    #         cur_root,
    #         spm_filename=spm_filename_prefix + ".model",
    #         yaml_filename=f"config_asr.yaml",
    #         specaugment_policy=None,
    #         extra={"use_audio_input": True}
    #     )
    # else:
    #     gen_config_yaml(
    #         cur_root,
    #         spm_filename=spm_filename_prefix + ".model",
    #         yaml_filename=f"config_asr.yaml",
    #         specaugment_policy="lb",
    #         cmvn_type=args.cmvn_type,
    #         gcmvn_path=(
    #             cur_root / "gcmvn.npz" if args.cmvn_type == "global"
    #             else None
    #         ),
    #     )
    # Clean up



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="char",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument(
        "--character-coverage",
        default=0.9995,
        required=True,
        type=float,
    ),
    parser.add_argument(
        "--cmvn-type", default="utterance",
        choices=["global", "utterance"],
        help="The type of cepstral mean and variance normalization"
    )
    parser.add_argument(
        "--gcmvn-max-num", default=150000, type=int,
        help="Maximum number of sentences to use to estimate global mean and "
             "variance"
        )
    parser.add_argument("--use-audio-input", action="store_true")
    args = parser.parse_args()


    Cantonese_ASR(args.data_root, "train", args)


if __name__ == "__main__":
    main()
