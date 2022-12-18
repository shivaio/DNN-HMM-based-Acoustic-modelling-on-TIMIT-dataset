import errno
import os
from os import path
import sys
import tarfile
import fnmatch
import pandas as pd
import subprocess
import argparse
from mapping import phone_maps
import python_speech_features as psf
import scipy.io.wavfile as wav
import numpy as np

timit_phone_map = phone_maps(mapping_file="kaldi_60_48_39.map")


def compute_mfcc(wav_file, n_delta=0):
    mfcc_feat = psf.mfcc(wav_file)
    if(n_delta == 0):
        return(mfcc_feat)
    elif(n_delta == 1):
        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1))))
    elif(n_delta == 2):
        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1), psf.delta(mfcc_feat, 2))))
    else:
        return 0

def read_transcript(full_wav):

    trans_file = full_wav[:-8] + ".PHN"
    with open(trans_file, "r") as file:
        trans = file.readlines()
    durations = [ele.strip().split(" ")[:-1] for ele in trans]
    durations_int = []
    for duration in durations:
        durations_int.append([int(duration[0]), int(duration[1])])
    trans = [ele.strip().split(" ")[-1] for ele in trans]
    trans = [timit_phone_map.map_symbol_reduced(symbol=phoneme) for phoneme in trans]
    return trans, durations_int

def _preprocess_data(args):
    target = args.TIMIT
    preprocessed = args.preprocessed
    print("Preprocessing data")
    full_wavs = []
    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*.WAV"):
            sph_file = os.path.join(root, filename)
            wav_file = os.path.join(root, filename)[:-4] + "_rif.wav"
            full_wavs.append(wav_file)
            print("converting {} to {}".format(sph_file, wav_file))
            if(~preprocessed):
                subprocess.check_call(["sox", sph_file, wav_file])

    print("Preprocessing Complete")
    print("Building features")

    mfcc_features = []
    mfcc_labels = []

    for full_wav in full_wavs:
        print("Computing features for file: ", full_wav)

        trans, durations = read_transcript(full_wav = full_wav)

        ndelta = int(args.ndelta)
        labels = []

        (sample_rate,wav_file) = wav.read(full_wav)

        mfcc_feats = compute_mfcc(wav_file[durations[0][0]:durations[0][1]], n_delta=ndelta)

        for i in range(len(mfcc_feats)):
            labels.append(trans[0])
        for index, chunk in enumerate(durations[1:]):
            mfcc_feat = compute_mfcc(wav_file[chunk[0]:chunk[1]], n_delta=ndelta)
            mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
            for i in range(len(mfcc_feat)):
                labels.append(trans[index])
        mfcc_features.extend(mfcc_feats)
        mfcc_labels.extend(labels)

    print(f" final mfcc features of entire dataset is {mfcc_features}")
    print(f" final labels of entire dataset is {mfcc_labels}")

    timit_df = pd.DataFrame()
    timit_df["features"] = mfcc_features
    timit_df["labels"] = mfcc_labels
    print(f" data frame of entire dataset is {timit_df}")
    timit_df.to_hdf("features/mfcc/timit.hdf", "timit")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TIMIT',type=str,default="/",help="Root directory containing timit dataset")
    parser.add_argument('--ndelta',type=str,default="0",help="Number of delta features to compute")
    parser.add_argument('--preprocessed',type=bool,default=False,help="Set to True if already preprocessed")
    args = parser.parse_args()
    _preprocess_data(args)
    print("Completed")