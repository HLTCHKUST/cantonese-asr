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



from fairseq.data.audio.audio_utils import get_waveform, convert_waveform


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]


class Cantonese_ASR(Dataset):
    """
    Create a Dataset for Cantonese_ASR. Each item is a tuple of the form:
    waveform, sample_rate, target utterance,
    utterance_id
    """

    SPLITS = ["train", "dev", "test"]


    def __init__(self, root: str,split: str) -> None:
        assert split in self.SPLITS 
        _root = Path(root) 
        wav_root, txt_root = _root / "audio", _root / "transcription"
        text_files=glob.glob('**/*.txt')
        wav_files=glob.glob('**/*.wav')
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files")
        # with open(txt_root / f"{split}.yaml") as f:
        tsv_file= _root/ 'test.csv'
        segments = pd.read_csv(tsv_file)

        # Load  target utterances
        utterances=[]
        for file in segments['text_path']:
            f= open(txt_root/file, 'r')
            utt=f.read()
            utterances.append(utt)
            
            # print(file, utt)
            # segments['text'] = utt
            # print(segments['text'])
            print(utterances)
        # assert len(segments) == len(utterances)
        
        # for i, u in enumerate(utterances):
        #     segments[i] = u
            
        # Gather info
        self.data = []
        for wav_filename, duration in zip(segments["audio_path"],segments["duration"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            # seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            # for i, segment in enumerate(seg_group):
            #     offset = int(float(segment["offset"]) * sample_rate)
            n_frames = int(float(duration) * sample_rate)
            _id = segments["id"]
            self.data.append(
                    (
                        wav_path.as_posix(),
                        n_frames,
                        _id,
                    )
                )

    def __getitem__(
            self, n: int
    ) -> Tuple[torch.Tensor, int,  str]:
        wav_path, tgt_utt, \
            utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, tgt_utt, utt_id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute()
    
    cur_root = Path(args.data_root) 
    if not cur_root.is_dir():
        print(f"{cur_root.as_posix()} does not exist. Skipped.")
    # Extract features
    audio_root = cur_root / ("flac" if args.use_audio_input else "fbank80")
    audio_root.mkdir(exist_ok=True)

    for split in Cantonese_ASR.SPLITS:
        print(f"Fetching split {split}...")
        dataset = Cantonese_ASR(root.as_posix(), split)
        if args.use_audio_input:
            print("Converting audios...")
            for waveform, _, utt_id in tqdm(dataset):
                tgt_sample_rate = 16_000
                _wavform, _ = convert_waveform(
                    waveform, sample_rate, to_mono=True,
                    to_sample_rate=tgt_sample_rate
                )
                sf.write(
                    (audio_root / f"{utt_id}.flac").as_posix(),
                    _wavform.numpy(), tgt_sample_rate
                )
        else:
            print("Extracting log mel filter bank features...")
            gcmvn_feature_list = []
            if split == 'train' and args.cmvn_type == "global":
                print("And estimating cepstral mean and variance stats...")

            for waveform, _, utt_id in tqdm(dataset):
                features = extract_fbank_features(
                    waveform,  audio_root / f"{utt_id}.npy"
                )
                if split == 'train' and args.cmvn_type == "global":
                    if len(gcmvn_feature_list) < args.gcmvn_max_num:
                        gcmvn_feature_list.append(features)

            if split == 'train' and args.cmvn_type == "global":
                # Estimate and save cmv
                stats = cal_gcmvn_stats(gcmvn_feature_list)
                with open(cur_root / "gcmvn.npz", "wb") as f:
                    np.savez(f, mean=stats["mean"], std=stats["std"])

    # Pack features into ZIP
    zip_path = cur_root / f"{audio_root.name}.zip"
    print("ZIPing audios/features...")
    create_zip(audio_root, zip_path)
    print("Fetching ZIP manifest...")
    audio_paths, audio_lengths = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    for split in Cantonese_ASR.SPLITS:
        is_train_split = split.startswith("train")
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = Cantonese_ASR(args.data_root, split)
        for _,  tgt_utt, utt_id in tqdm(dataset):
            manifest["id"].append(utt_id)
            manifest["audio"].append(audio_paths[utt_id])
            manifest["n_frames"].append(audio_lengths[utt_id])
            manifest["tgt_text"].append(
                tgt_utt
            )
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, cur_root / f"{split}_asr.tsv")
    # Generate vocab
    v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_asr"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            cur_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
        )
    # Generate config YAML
    if args.use_audio_input:
        gen_config_yaml(
            cur_root,
            spm_filename=spm_filename_prefix + ".model",
            yaml_filename=f"config_asr.yaml",
            specaugment_policy=None,
            extra={"use_audio_input": True}
        )
    else:
        gen_config_yaml(
            cur_root,
            spm_filename=spm_filename_prefix + ".model",
            yaml_filename=f"config_asr.yaml",
            specaugment_policy="lb",
            cmvn_type=args.cmvn_type,
            gcmvn_path=(
                cur_root / "gcmvn.npz" if args.cmvn_type == "global"
                else None
            ),
        )
    # Clean up
    shutil.rmtree(audio_root)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
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


    process(args)


if __name__ == "__main__":
    main()
