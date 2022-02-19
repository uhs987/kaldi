#!/usr/bin/env bash

set -e

stage=0
do_test_sets_mfcc=false
do_multi_cn_lm_test=false
do_ngram_lm_test=false
do_rnn_lm_test=false
do_pytorch_transformer_lm_test=false
do_pytorch_rnn_lstm_lm_test=false
do_pytorch_rnn_gru_lm_test=false

# To be run from one directory above this script.
. ./cmd.sh
. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [options] <lm-name> <lm-corpus-path>"
  echo " Options:"
  echo "  --do-test-sets-mfcc : If true, do mfcc for test corpora"
  echo "  --do_multi-cn-lm-test : If true, decode test_my_1 with multi_cn's LM only"
  echo "  --do-ngram-lm-test : If true, do ngram LM test"
  echo "  --do-rnn-lm-test : If true, do rnn LM test"
  echo "  --do-pytorch-transformer-lm-test : If true, do rnn LM test"
  echo "  --do-pytorch-rnn-lstm-lm-test : If true, do rnn LM test"
  echo "  --do-pytorch-rnn-gru-lm-test : If true, do rnn LM test"

  echo "  e.g.: $0 test_my_2 data/military_fans/test/text"
  echo "      : $0 test_my_3 /work/u7438383/data/lm-corpus/lm-corpus-2021-1118.txt"
  echo "      : $0 test_my_4 /work/u7438383/data/lm-corpus/lm-corpus-2021-1211-total.txt"
  echo "      : $0 test_my_5 /work/u7438383/data/lm-corpus/lm-corpus-2021-1214-total.txt"
  echo "      : $0 test_my_6 /work/u7438383/data/lm-corpus/lm-corpus-2021-1228-total.txt"

  exit 1;
fi

lm_name=$1
lm_text=$2

test_sets="10test"

if $do_test_sets_mfcc; then
  if [ $stage -le 0 ]; then
    # Now make MFCC plus pitch features.
    # mfccdir should be some place with a largish disk where you
    # want to store MFCC features.
    mfccdir=mfcc

    echo "$0: creating MFCC features"
    for c in $test_sets; do
      steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 10 \
        data/$c/test exp/make_mfcc/$c/test $mfccdir/$c || exit 1;
      steps/compute_cmvn_stats.sh data/$c/test \
        exp/make_mfcc/$c/test $mfccdir/$c || exit 1;
      utils/fix_data_dir.sh data/$c/test || exit 1;
    done
  fi

  # from local/chain/run_ivector_common.sh
  nnet3_affix=_cleaned

  if [ $stage -le 1 ]; then
    echo "$0: creating high-resolution MFCC features"
    for c in $test_sets; do
      utils/copy_data_dir.sh data/$c/test data/$c/test_hires

      steps/make_mfcc.sh --nj 10 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/$c/test_hires || exit 1;
      steps/compute_cmvn_stats.sh data/$c/test_hires || exit 1;
      utils/fix_data_dir.sh data/$c/test_hires
    done
  fi

  if [ $stage -le 2 ]; then
    echo "$0: extracting iVectors for test data"
    for c in $test_sets; do
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 10 \
        data/${c}/test_hires exp/nnet3${nnet3_affix}/extractor \
        exp/nnet3${nnet3_affix}/ivectors_${c}_hires || exit 1;
    done
  fi
fi

if $do_multi_cn_lm_test; then
  if [ $stage -le 3 ]; then
    echo "$0: run multi-cn lm test, lm-type origin"
    local/lm_test.sh --lm-type origin --test-sets "$test_sets" test_my_1 data/train_combined/text
  fi

  if [ $stage -le 4 ]; then
    echo "$0: run RNN lm test, lm-type origin"
    local/rnnlm/run_tdnn_lstm.sh --lm-type origin --test-sets "$test_sets" test_my_1
  fi
fi

if $do_ngram_lm_test; then
  if [ $stage -le 5 ]; then
    echo "$0: run n-gram lm test, lm-type nointerp"
    local/lm_test.sh --lm-type nointerp --test-sets "$test_sets" $lm_name $lm_text
  fi

  if [ $stage -le 6 ]; then
    echo "$0: run n-gram lm test, lm-type nointerp-lexicon"
    local/lm_test.sh --lm-type nointerp-lexicon --test-sets "$test_sets" $lm_name $lm_text
  fi

  if [ $stage -le 7 ]; then
    echo "$0: run n-gram lm test, lm-type interp"
    local/lm_test.sh --lm-type interp --test-sets "$test_sets" $lm_name $lm_text
  fi

  if [ $stage -le 8 ]; then
    echo "$0: run n-gram lm test, lm-type interp-lexicon"
    local/lm_test.sh --lm-type interp-lexicon --test-sets "$test_sets" $lm_name $lm_text
  fi
fi

if $do_rnn_lm_test; then
  # use py36
  if [ $stage -le 9 ]; then
    echo "$0: run RNN lm test, lm-type nointerp"
    local/rnnlm/run_tdnn_lstm.sh --lm-type nointerp --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 10 ]; then
    echo "$0: run RNN lm test, lm-type nointerp-lexicon"
    local/rnnlm/run_tdnn_lstm.sh --lm-type nointerp-lexicon --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 11 ]; then
    echo "$0: run RNN lm test, lm-type interp"
    local/rnnlm/run_tdnn_lstm.sh --lm-type interp --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 12 ]; then
    echo "$0: run RNN lm test, lm-type interp-lexicon"
    local/rnnlm/run_tdnn_lstm.sh --lm-type interp-lexicon --test-sets "$test_sets" $lm_name
  fi
fi

if $do_pytorch_transformer_lm_test; then
  if [ $stage -le 13 ]; then
    echo "$0: run pytorch transformer lm test, lm-type nointerp"
    local/pytorchnn/run_nnlm.sh --lm-type nointerp --model-type Transformer --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 14 ]; then
    echo "$0: run pytorch transformer lm test, lm-type nointerp-lexicon"
    local/pytorchnn/run_nnlm.sh --lm-type nointerp-lexicon --model-type Transformer --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 15 ]; then
    echo "$0: run pytorch transformer lm test, lm-type interp"
    local/pytorchnn/run_nnlm.sh --lm-type interp --model-type Transformer --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 16 ]; then
    echo "$0: run pytorch transformer lm test, lm-type interp-lexicon"
    local/pytorchnn/run_nnlm.sh --lm-type interp-lexicon --model-type Transformer --test-sets "$test_sets" $lm_name
  fi
fi

if $do_pytorch_rnn_lstm_lm_test; then
  if [ $stage -le 17 ]; then
    echo "$0: run pytorch RNN LSTM lm test, lm-type nointerp"
    local/pytorchnn/run_nnlm.sh --lm-type nointerp --model-type LSTM --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 18 ]; then
    echo "$0: run pytorch RNN LSTM lm test, lm-type nointerp-lexicon"
    local/pytorchnn/run_nnlm.sh --lm-type nointerp-lexicon --model-type LSTM --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 19 ]; then
    echo "$0: run pytorch RNN LSTM lm test, lm-type interp"
    local/pytorchnn/run_nnlm.sh --lm-type interp --model-type LSTM --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 20 ]; then
    echo "$0: run pytorch RNN LSTM lm test, lm-type interp-lexicon"
    local/pytorchnn/run_nnlm.sh --lm-type interp-lexicon --model-type LSTM --test-sets "$test_sets" $lm_name
  fi
fi

if $do_pytorch_rnn_gru_lm_test; then
  if [ $stage -le 21 ]; then
    echo "$0: run pytorch RNN GRU lm test, lm-type nointerp"
    local/pytorchnn/run_nnlm.sh --lm-type nointerp --model-type GRU --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 22 ]; then
    echo "$0: run pytorch RNN GRU lm test, lm-type nointerp-lexicon"
    local/pytorchnn/run_nnlm.sh --lm-type nointerp-lexicon --model-type GRU --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 23 ]; then
    echo "$0: run pytorch RNN GRU lm test, lm-type interp"
    local/pytorchnn/run_nnlm.sh --lm-type interp --model-type GRU --test-sets "$test_sets" $lm_name
  fi

  if [ $stage -le 24 ]; then
    echo "$0: run pytorch RNN GRU lm test, lm-type interp-lexicon"
    local/pytorchnn/run_nnlm.sh --lm-type interp-lexicon --model-type GRU --test-sets "$test_sets" $lm_name
  fi
fi

exit 0
