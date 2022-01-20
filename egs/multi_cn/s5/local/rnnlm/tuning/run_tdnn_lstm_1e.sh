#!/usr/bin/env bash

# This script is copied from swbd/s5c

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

ac_model_dir=exp/chain_cleaned/tdnn_cnn_1a_sp
decode_dir_suffix=rnnlm_1e
ngram_order=3 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially
pruned_rescore=true

lm_type=
test_sets=""

. ./cmd.sh
. ./utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <lm-name>"
  echo " Options:"
  echo "  --lm-type [origin|nointerp|interp] : type of language model to create"
  echo "  --test-sets : test sets to be decoded"

  exit 1;
fi

lm_name=$1

lm_text_only=false
lm_lexicon=false

case $lm_type in
  origin)
    lm_text_only=true
    lm_suffix=""
    LM=combined_tg
    ;;
  nointerp)
    lm_text_only=true
    lm_suffix=""
    LM=${lm_name}_tg
    ;;
  nointerp-lexicon)
    lm_text_only=true
    lm_lexicon=true
    lm_suffix="_lexicon"
    LM=${lm_name}${lm_suffix}_tg
    ;;
  interp)
    lm_suffix="_interp"
    LM=${lm_name}${lm_suffix}_tg
    ;;
  interp-lexicon)
    lm_lexicon=true
    lm_suffix="_interp_lexicon"
    LM=${lm_name}${lm_suffix}_tg
    ;;
  *)
    echo "Invalid --lmtype option: $lm_type"
    exit 1
    ;;
esac

text=data/train_combined/text
lm_text=data/local/lm/text.$lm_name
if $lm_lexicon; then
  words=data/lang/${lm_name}${lm_suffix}/words.txt
else
  words=data/lang/words.txt
fi
text_dir=data/rnnlm/text_1e/${lm_name}${lm_suffix}
dir=$dir/${lm_name}${lm_suffix}
mkdir -p $dir/config
set -e

for f in $text $lexicon; do
  [ ! -f $f ] && \
    echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 0 ]; then
  mkdir -p $text_dir
  echo -n >$text_dir/dev.txt
  # hold out one in every 50 lines as dev data.
  if $lm_text_only; then
    cat $lm_text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/lm.txt
  else
    cat $text | cut -d ' ' -f2- | awk -v text_dir=$text_dir '{if(NR%50 == 0) { print >text_dir"/dev.txt"; } else {print;}}' >$text_dir/multi_cn.txt

    cat $lm_text | cut -d ' ' -f2- | awk '{print;}' >$text_dir/lm.txt
  fi
fi

if [ $stage -le 1 ]; then
  cp $words $dir/config/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $dir/config/words.txt

  # words that are not present in words.txt but are in the training or dev data, will be
  # mapped to <SPOKEN_NOISE> during training.
  echo "<UNK>" >$dir/config/oov.txt

  if $lm_text_only; then
    cat > $dir/config/data_weights.txt <<EOF
lm   1   1.0
EOF
  else
    cat > $dir/config/data_weights.txt <<EOF
multi_cn   3   1.0
lm   1   1.0
EOF
  fi

  rnnlm/get_unigram_probs.py --vocab-file=$dir/config/words.txt \
                             --unk-word="<UNK>" \
                             --data-weights-file=$dir/config/data_weights.txt \
                             $text_dir | awk 'NF==2' >$dir/config/unigram_probs.txt

  # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --special-words='<s>,</s>,<brk>,<UNK>,<eps>,!SIL' \
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
  echo "$0: Preparing rnnlm directory $dir"
  rnnlm/prepare_rnnlm_dir.sh $text_dir $dir/config $dir
fi

if [ $stage -le 3 ]; then
  echo "$0: Training RNNLM model"
  rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 --use-gpu wait \
                  --stage $train_stage --num-epochs 10 --cmd "$train_cmd" $dir
fi

if [ $stage -le 4 ] && $run_lat_rescore; then
  echo "$0: Perform lattice-rescoring on $ac_model_dir"
  pruned=
  if $pruned_rescore; then
    pruned=_pruned
  fi
  for decode_set in $test_sets; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}

    # Lattice rescoring
    rnnlm/lmrescore$pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.45 --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/${decode_set}/test_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}_0.45
  done
fi

if [ $stage -le 5 ] && $run_nbest_rescore; then
  echo "$0: Perform nbest-rescoring on $ac_model_dir"
  for decode_set in $test_sets; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}

    # Lattice rescoring
    rnnlm/lmrescore_nbest.sh \
      --cmd "$decode_cmd --mem 4G" --N 20 \
      0.8 data/lang_$LM $dir \
      data/${decode_set}/test_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix}_nbest
  done
fi

if [ $stage -le 8 ]; then
  echo "$lm_name results:"

  for c in $test_sets; do
    echo "$c:"
    echo "--- CER ---"
    for x in exp/*/*/decode_${c}_${LM}_${decode_dir_suffix}_0.45; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    for x in exp/*/*/decode_${c}_${LM}_${decode_dir_suffix}_nbest; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    echo "--- WER ---"
    for x in exp/*/*/decode_${c}_${LM}_${decode_dir_suffix}_0.45; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
    for x in exp/*/*/decode_${c}_${LM}_${decode_dir_suffix}_nbest; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
  done

  echo ""
fi

exit 0
