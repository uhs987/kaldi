#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2015  Guoguo Chen
#           2017  Hainan Xu
#           2017  Xiaohui Zhang

# This script trains a backward LMs on the swbd LM-training data, and use it
# to rescore either decoded lattices, or lattices that are just rescored with
# a forward RNNLM. In order to run this, you must first run the forward RNNLM
# recipe at local/rnnlm/run_tdnn_lstm.sh

# rnnlm/train_rnnlm.sh: best iteration (out of 10) was 7, linking it to final iteration.
# rnnlm/train_rnnlm.sh: train/dev perplexity was 129.8 / 338.3.
# Train objf: -6.37 -5.88 -5.59 -5.36 -5.16 -4.98 -4.82 -4.67 -4.52
# Dev objf:   -6.95 -6.27 -6.04 -5.92 -5.86 -5.83 -5.82 -5.84 -5.86

# %WER 6.19 [ 6483 / 104765, 250 ins, 432 del, 5801 sub ] exp/chain/tdnn_1a_sp/decode_test_rnnlm_1e_back_0.45/cer_10_1.0

# %WER 14.65 [ 9441 / 64428, 759 ins, 1531 del, 7151 sub ] exp/chain/tdnn_1a_sp/decode_test_rnnlm_1e_back_0.45/wer_12_1.0

# Begin configuration section.

dir=exp/rnnlm_lstm_1e_backward
embedding_dim=1024
lstm_rpd=256
lstm_nrpd=256
stage=-10
train_stage=-10

# variables for lattice rescoring
run_lat_rescore=true
ac_model_dir=exp/chain/tdnn_1a_sp
decode_dir_suffix_forward=rnnlm_1e
decode_dir_suffix_backward=rnnlm_1e_back
ngram_order=3 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially

. ./cmd.sh
. ./utils/parse_options.sh

text=data/local/train/text
lexicon=data/local/dict/lexiconp.txt
text_dir=data/rnnlm/text_1e_back
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
  cat $text | cut -d ' ' -f2- | awk '{for(i=NF;i>0;i--) printf("%s ", $i); print""}' | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/aishell.txt
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

  for decode_set in test; do
    decode_dir=${ac_model_dir}/decode_${decode_set}
    if [ ! -d ${decode_dir}_${decode_dir_suffix_forward}_0.45 ]; then
      echo "$0: Must run the forward recipe first at local/rnnlm/run_tdnn_lstm.sh"
      exit 1
    fi

    # Lattice rescoring
    rnnlm/lmrescore_back.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.45 --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/${decode_set}_hires ${decode_dir}_${decode_dir_suffix_forward}_0.45 \
      ${decode_dir}_${decode_dir_suffix_backward}_0.45
  done
fi

if [ $stage -le 6 ]; then
  echo "RNNLM backward decode results:"

  for c in test; do
    echo "$c:"
    echo "--- CER ---"
    for x in $ac_model_dir/decode_${c}_${decode_dir_suffix_backward}_0.45; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    echo "--- WER ---"
    for x in $ac_model_dir/decode_${c}_${decode_dir_suffix_backward}_0.45; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
  done

  echo ""
fi

exit 0
