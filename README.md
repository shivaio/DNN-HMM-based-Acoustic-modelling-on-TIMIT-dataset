# DNN-HMM-based-Acoustic-modelling-on-TIMIT-dataset

Speech recognition based on deep neural network/hidden markov model:

1. Extracted MFCC features from each frame of phoneme.
2. Perform the GMM/HMM based Viterbi algorithm.
3. Prepare unique HMM state IDs.
Use this unique HMM state ID to convert the all state sequence obtained in the step 2.

DNN training:

1. Set the DNN topologies.
2. Perform the DNN training.

Predict the most likely digit for each utterance by selecting the largest likelihood digit.

Compute the accuracy (# of correct digits / # of test utterances * 100) by using whole training data.
