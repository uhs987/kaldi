#!/usr/bin/env bash

set -e

stage=0
lm_type=
test_sets=""

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [options] <lm-name> <lm-text>"
  echo " Options:"
  echo "  --lm-type [origin|nointerp|interp] : type of language model to create"
  echo "  --test-sets : test sets to be decoded"

  exit 1;
fi

lm_name=$1
lm_text=$2

case $lm_type in
  origin)
    lm_suffix=""
    LM=combined
    ;;
  nointerp)
    lm_suffix=""
    LM=${lm_name}
    ;;
  nointerp-lexicon)
    lm_suffix="_lexicon"
    LM=${lm_name}${lm_suffix}
    ;;
  interp)
    lm_suffix="_interp"
    LM=${lm_name}${lm_suffix}
    ;;
  interp-lexicon)
    lm_suffix="_interp_lexicon"
    LM=${lm_name}${lm_suffix}
    ;;
  *)
    echo "Invalid --lm-type option: $lm_type"
    exit 1
    ;;
esac

if [ $stage -le 0 ]; then
  echo "$0: creating language models"

  if [[ $lm_type == "origin" ]]; then
    # It seems redundant to train the LM again but the decode result will
    # be incorrect without doing this.
    #
    # TODO: investigate the root cause.
    cp data/train_combined/text data/local/lm/text.$1

    # LM: train_combined text, data/local/lm
    local/train_lms.sh || exit 1;

    # prepare LM
    utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
      data/local/dict/lexicon.txt data/lang_${LM}_tg || exit 1;
    utils/format_lm.sh data/lang data/local/lm/4gram-mincount/lm_unpruned.gz \
      data/local/dict/lexicon.txt data/lang_${LM}_fg || exit 1;
  elif [[ $lm_type == "nointerp" ]]; then
    cp $lm_text data/local/lm/text.$1

    # LM: LM text, data/local/lm/$lm_name
    local/train_lms.sh --lm-text data/local/lm/text.$lm_name $lm_name || exit 1;

    # G compilation, check LG composition
    utils/format_lm.sh data/lang data/local/lm/$lm_name/3gram-mincount/lm_unpruned.gz \
      data/local/dict/lexicon.txt data/lang_${LM}_tg || exit 1;
    utils/format_lm.sh data/lang data/local/lm/$lm_name/4gram-mincount/lm_unpruned.gz \
      data/local/dict/lexicon.txt data/lang_${LM}_fg || exit 1;
  elif [[ $lm_type == "nointerp-lexicon" ]]; then
    cp $lm_text data/local/lm/text.$lm_name

    # dict: train_combined+test_combined+LM text, data/local/dict/$lm_name
    local/prepare_dict.sh --lm-text data/local/lm/text.$lm_name $lm_name
    local/fix_lm_dict.py $lm_name

    # LM: LM text, data/local/lm/${lm_name}${lm_suffix}
    local/train_lms.sh --lm-text data/local/lm/text.$lm_name --lexicon-text data/local/dict/$lm_name/lexicon.txt \
      ${lm_name}${lm_suffix} || exit 1;

    utils/prepare_lang.sh data/local/dict/${lm_name} "<UNK>" data/local/lang/${lm_name}${lm_suffix} data/lang/${lm_name}${lm_suffix}

    # G compilation, check LG composition
    utils/format_lm.sh data/lang/${lm_name}${lm_suffix} data/local/lm/${lm_name}${lm_suffix}/3gram-mincount/lm_unpruned.gz \
      data/local/dict/${lm_name}/lexicon.txt data/lang_${LM}_tg || exit 1;
    utils/format_lm.sh data/lang/${lm_name}${lm_suffix} data/local/lm/${lm_name}${lm_suffix}/4gram-mincount/lm_unpruned.gz \
      data/local/dict/${lm_name}/lexicon.txt data/lang_${LM}_fg || exit 1;
  elif [[ $lm_type == "interp" ]]; then
    cp $lm_text data/local/lm/text.$1

    # LM: LM text, data/local/lm/${lm_name}${lm_suffix}
    local/train_corpus_lm.sh --lm-text data/local/lm/text.$lm_name \
      ${lm_name}${lm_suffix} data/local/lm/3gram-mincount/lm_unpruned.gz data/local/lm/4gram-mincount/lm_unpruned.gz || exit 1;

    # lexicon unchanged so skip the prepare_lang.sh

    # G compilation, check LG composition
    utils/format_lm.sh data/lang data/local/lm/${lm_name}${lm_suffix}/lm_interp.gz \
      data/local/dict/lexicon.txt data/lang_${LM}_tg || exit 1;
    utils/format_lm.sh data/lang data/local/lm/${lm_name}${lm_suffix}/lm_interp_fg.gz \
      data/local/dict/lexicon.txt data/lang_${LM}_fg || exit 1;
  elif [[ $lm_type == "interp-lexicon" ]]; then
    cp $lm_text data/local/lm/text.$lm_name

    # dict: train_combined+test_combined+LM text, data/local/dict/$lm_name
    local/prepare_dict.sh --lm-text data/local/lm/text.$lm_name $lm_name
    local/fix_lm_dict.py $lm_name

    # LM: LM text, data/local/lm/${lm_name}${lm_suffix}
    local/train_corpus_lm.sh --lm-text data/local/lm/text.$lm_name --lexicon-text data/local/dict/$lm_name/lexicon.txt \
      ${lm_name}${lm_suffix} data/local/lm/3gram-mincount/lm_unpruned.gz data/local/lm/4gram-mincount/lm_unpruned.gz|| exit 1;

    utils/prepare_lang.sh data/local/dict/${lm_name} "<UNK>" data/local/lang/${lm_name}${lm_suffix} data/lang/${lm_name}${lm_suffix}

    # G compilation, check LG composition
    utils/format_lm.sh data/lang/${lm_name}${lm_suffix} data/local/lm/${lm_name}${lm_suffix}/lm_interp.gz \
      data/local/dict/${lm_name}/lexicon.txt data/lang_${LM}_tg || exit 1;
    utils/format_lm.sh data/lang/${lm_name}${lm_suffix} data/local/lm/${lm_name}${lm_suffix}/lm_interp_fg.gz \
      data/local/dict/${lm_name}/lexicon.txt data/lang_${LM}_fg || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  # decode tri1b
  echo "$0: decoding tri1b"
  graph_dir=exp/tri1b/graph_${lm_name}${lm_suffix}_tg
  utils/mkgraph.sh data/lang_${LM}_tg exp/tri1b $graph_dir || exit 1;

  for c in $test_sets; do
    decode_dir=exp/tri1b/decode_${c}_${LM}

    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
      $graph_dir data/$c/test ${decode_dir}_tg || exit 1;
  done
fi

if [ $stage -le 2 ]; then
  # decode tri2a
  echo "$0: decoding tri2"
  graph_dir=exp/tri2a/graph_${lm_name}${lm_suffix}_tg
  utils/mkgraph.sh data/lang_${LM}_tg exp/tri2a $graph_dir || exit 1;

  for c in $test_sets; do
    decode_dir=exp/tri2a/decode_${c}_${LM}

    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
      $graph_dir data/$c/test ${decode_dir}_tg || exit 1;
  done
fi

if [ $stage -le 3 ]; then
  # decode tri3a
  echo "$0: decoding tri3a"
  graph_dir=exp/tri3a/graph_${lm_name}${lm_suffix}_tg
  utils/mkgraph.sh data/lang_${LM}_tg exp/tri3a $graph_dir || exit 1;

  for c in $test_sets; do
    decode_dir=exp/tri3a/decode_${c}_${LM}

    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
      $graph_dir data/$c/test ${decode_dir}_tg || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  # decode tri4a
  echo "$0: decoding tri4a"
  graph_dir=exp/tri4a/graph_${lm_name}${lm_suffix}_tg
  utils/mkgraph.sh data/lang_${LM}_tg exp/tri4a $graph_dir || exit 1;

  for c in $test_sets; do
    decode_dir=exp/tri4a/decode_${c}_${LM}

    steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
      $graph_dir data/$c/test ${decode_dir}_tg || exit 1;
  done
fi

# from local/run_cleanup_segmentation.sh
cleanup_affix=cleaned
srcdir=exp/tri4a
decode_nj=10
decode_num_threads=4
cleaned_dir=${srcdir}_${cleanup_affix}

if [ $stage -le 5 ]; then
  # Test with the models trained on cleaned-up data.
  echo "$0: decoding tri4a_cleaned"
  graph_dir=${cleaned_dir}/graph_${lm_name}${lm_suffix}_tg
  utils/mkgraph.sh data/lang_${LM}_tg ${cleaned_dir} $graph_dir

  for c in $test_sets; do
    decode_dir=${cleaned_dir}/decode_${c}_${LM}

    steps/decode_fmllr.sh --nj $decode_nj --num-threads $decode_num_threads \
      --cmd "$decode_cmd" \
      $graph_dir data/$c/test ${decode_dir}_tg
  done
fi

# from local/chain/tuning/run_cnn_tdnn_1a.sh
decode_nj=10
nnet3_affix=_cleaned
affix=cnn_1a
dir=exp/chain${nnet3_affix}/tdnn${affix:+_$affix}_sp
graph_dir=$dir/graph_${lm_name}${lm_suffix}_tg
if [ $stage -le 6 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov \
    data/lang_${LM}_tg $dir $graph_dir
  # remove <UNK> (word id is 3) from the graph, and convert back to const-FST.
  oom_int=$(cat data/lang_${LM}_tg/oov.int)
  fstrmsymbols --apply-to-output=true --remove-arcs=true "echo ${oom_int}|" $graph_dir/HCLG.fst - | \
    fstconvert --fst_type=const > $graph_dir/temp.fst
  mv $graph_dir/temp.fst $graph_dir/HCLG.fst
fi

if [ $stage -le 7 ]; then
  echo "$0: decoding tdnn${affix:+_$affix}_sp"
  rm $dir/.error 2>/dev/null || true
  for c in $test_sets; do
    decode_dir=$dir/decode_${c}_${LM}

    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $decode_nj --cmd "$decode_cmd" \
      --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${c}_hires \
      $graph_dir data/$c/test_hires ${decode_dir}_tg || exit 1

    # 4gram-LM rescore
    steps/lmrescore.sh --cmd "$decode_cmd" \
      --self-loop-scale 1.0 \
      data/lang_${LM}_tg data/lang_${LM}_fg \
      data/$c/test_hires ${decode_dir}_tg ${decode_dir}_fg || exit 1

    # for pytorch lattice rescore
    for n in 1 2 3 4 5 6 7 8 9 10; do
      decode_dir=$dir/decode_${c}-${n}_${LM}
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $decode_nj --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${c}_hires \
        $graph_dir data/$c/test_hires-$n ${decode_dir}_tg || exit 1
    done
  done
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if [ $stage -le 8 ]; then
  echo "$lm_name results:"

  for c in $test_sets; do
    echo "$c:"
    echo "--- CER ---"
    for x in exp/*/decode_${c}_${lm_name}${lm_suffix}_tg; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    for x in exp/*/*/decode_${c}_${lm_name}${lm_suffix}_{tg,fg}; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    for n in 1 2 3 4 5 6 7 8 9 10; do
      for x in exp/*/*/decode_${c}-${n}_${lm_name}${lm_suffix}_tg; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    done
    echo "--- WER ---"
    for x in exp/*/decode_${c}_${lm_name}${lm_suffix}_tg; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
    for x in exp/*/*/decode_${c}_${lm_name}${lm_suffix}_{tg,fg}; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
    for n in 1 2 3 4 5 6 7 8 9 10; do
      for x in exp/*/*/decode_${c}-${n}_${lm_name}${lm_suffix}_tg; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
    done
  done

  echo ""
fi
