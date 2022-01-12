#!/usr/bin/env bash

set -e

lm_names="test_my_6"
test_sets="10test"

. ./utils/parse_options.sh || exit 1;

for m in $lm_names; do
  echo "LM: $m"

  for c in $test_sets; do
    echo "Test set: $c"
    decode_dir=decode_${c}_${m}
    decode_dir_suffixs=("" _lexicon _interp _interp_lexicon)

    echo "--- CER ---"
    for decode_dir_suffix in "${decode_dir_suffixs[@]}"; do
      echo "- base -"
      for x in exp/{tri1b,tri2a,tri3a,tri4a,tri4a_cleaned}/${decode_dir}${decode_dir_suffix}_tg; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
      for x in exp/*/*/${decode_dir}${decode_dir_suffix}_tg; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
      # 4gram
      echo "- 4gram rescore -"
      for x in exp/*/*/${decode_dir}${decode_dir_suffix}_fg; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
      # nbest
      echo "- nbest rescore -"
      for x in exp/*/*/${decode_dir}${decode_dir_suffix}_tg_rnnlm_1e_nbest; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
      for x in exp/*/*/${decode_dir}${decode_dir_suffix}_tg_pytorch_{transformer,lstm}_nbest*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
      # lattice
      echo "- lattice rescore -"
      for x in exp/*/*/${decode_dir}${decode_dir_suffix}_tg_rnnlm_1e_0.45; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
      for p in transformer lstm; do
        for n in 1 2 3 4 5 6 7 8 9 10; do
          decode_dir_n=decode_${c}-${n}_${m}
          for x in exp/*/*/${decode_dir_n}${decode_dir_suffix}_tg_pytorch_${p}*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
        done
      done
      echo ""
    done

    echo "--- WER ---"
    for decode_dir_suffix in "${decode_dir_suffixs[@]}"; do
      echo "- base -"
      for x in exp/{tri1b,tri2a,tri3a,tri4a,tri4a_cleaned}/${decode_dir}${decode_dir_suffix}_tg; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
      for x in exp/*/*/${decode_dir}${decode_dir_suffix}_tg; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
      # 4gram
      echo "- 4gram rescore -"
      for x in exp/*/*/${decode_dir}${decode_dir_suffix}_fg; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
      # nbest
      echo "- nbest rescore -"
      for x in exp/*/*/${decode_dir}${decode_dir_suffix}_tg_rnnlm_1e_nbest; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
      for x in exp/*/*/${decode_dir}${decode_dir_suffix}_tg_pytorch_{transformer,lstm}_nbest*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
      # lattice
      echo "- lattice rescore -"
      for x in exp/*/*/${decode_dir}${decode_dir_suffix}_tg_rnnlm_1e_0.45; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
      for p in transformer lstm; do
        for n in 1 2 3 4 5 6 7 8 9 10; do
          decode_dir_n=decode_${c}-${n}_${m}
          for x in exp/*/*/${decode_dir_n}${decode_dir_suffix}_tg_pytorch_${p}*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
        done
      done
      echo ""
    done
  done

  echo ""
done

exit 0;
