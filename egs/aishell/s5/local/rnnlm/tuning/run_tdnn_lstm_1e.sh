#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Xiaohui Zhang

# This script trains LMs on the aishell LM-training data.

# rnnlm/train_rnnlm.sh: best iteration (out of 10) was 6, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 146.5 / 347.9.
# Train objf: -6.41 -5.89 -5.59 -5.33 -5.12 -4.93 -4.77 -4.62 -4.46
# Dev objf:   -6.99 -6.30 -6.04 -5.93 -5.87 -5.85 -5.87 -5.88 -5.92

# %WER 6.25 [ 6553 / 104765, 242 ins, 389 del, 5922 sub ] exp/chain/tdnn_1a_sp/decode_test_rnnlm_1e_0.45/cer_10_0.5
# %WER 6.43 [ 6733 / 104765, 272 ins, 451 del, 6010 sub ] exp/chain/tdnn_1a_sp/decode_test_rnnlm_1e_nbest/cer_11_1.0

# %WER 14.39 [ 9270 / 64428, 772 ins, 1494 del, 7004 sub ] exp/chain/tdnn_1a_sp/decode_test_rnnlm_1e_0.45/wer_12_0.5
# %WER 14.95 [ 9632 / 64428, 1071 ins, 1247 del, 7314 sub ] exp/chain/tdnn_1a_sp/decode_test_rnnlm_1e_nbest/wer_13_0.0

# Begin configuration section.

dir=exp/rnnlm_lstm_1e
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=-10
train_stage=-10

# variables for lattice rescoring
run_lat_rescore=true
run_nbest_rescore=true
run_backward_rnnlm=false

ac_model_dir=exp/chain/tdnn_1a_sp
decode_dir_suffix=rnnlm_1e
ngram_order=3 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

. ./cmd.sh
. ./utils/parse_options.sh

text=data/local/train/text
lexicon=data/local/dict/lexiconp.txt
text_dir=data/rnnlm/text_1e
mkdir -p $dir/config
set -e

for f in $text $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist; search for local/wsj_extend_dict.sh in run.sh" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 50 lines as dev data.
  cat $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/aishell.txt
fi

if [ $stage -le 1 ]; then
  cp data/lang/words.txt $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<SPOKEN_NOISE>" >$dir/config/oov.txt

  cat > $dir/config/data_weights.txt <<EOF
aishell   3   1.0
EOF

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<SPOKEN_NOISE>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<brk>,<SPOKEN_NOISE>,<eps>,SIL' \
                           $dir/config/words.txt > $dir/config/features.txt

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim input=Append(0, IfDefined(-1))
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
relu-renorm-layer name=tdnn2 dim=$embedding_dim input=Append(0, IfDefined(-3))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd
relu-renorm-layer name=tdnn3 dim=$embedding_dim input=Append(0, IfDefined(-3))
output-layer name=output include-log-softmax=false dim=$embedding_dim
EOF
  rnnlm/validate_config_dir.sh $text_dir $dir/config
fi

if [ $stage -le 2 ]; then
  echo "$0: Prepare rnnlm directory $dir"
  rnnlm/prepare_rnnlm_dir.sh $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  echo "$0: Train RNNLM model"
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 --use-gpu wait \
                  --stage $train_stage --num-epochs 10 --cmd "$train_cmd" $dir
fi

LM=test
if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  for decode_set in test; do
    decode_dir=${ac_model_dir}/decode_${decode_set}

    # Lattice rescoring
    rnnlm/lmrescore$pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.45 --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}_0.45
  done
fi

if [ $stage -le 5 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in test; do
    decode_dir=${ac_model_dir}/decode_${decode_set}

    # Lattice rescoring
    rnnlm/lmrescore_nbest.sh \
      --cmd "$decode_cmd --mem 4G" --N 20 \
      0.8 data/lang_$LM $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}_nbest
  done
fi

if [ $stage -le 6 ]; then
  echo "RNNLM decode results:"

  for c in test; do
    echo "$c:"
    echo "--- CER ---"
    for x in $ac_model_dir/decode_${c}_${decode_dir_suffix}_0.45; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    for x in $ac_model_dir/decode_${c}_${decode_dir_suffix}_nbest; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    echo "--- WER ---"
    for x in $ac_model_dir/decode_${c}_${decode_dir_suffix}_0.45; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
    for x in $ac_model_dir/decode_${c}_${decode_dir_suffix}_nbest; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
  done

  echo ""
fi

# running backward RNNLM, which further improves WERS by combining backward with
# the forward RNNLM trained in this script.
if [ $stage -le 7 ] && $run_backward_rnnlm; then
  local/rnnlm/run_tdnn_lstm_back.sh
fi

exit 0
