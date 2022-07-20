#!/usr/bin/env bash

set -e

stage=0
decode_root="./exp/chain/tdnn_1a_sp"
test_set="test"
#weight_function="ppl-count"
#weight_function="ppl"
weight_function="count"

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;

# begin configuration section.
cmd=run.pl

lm_tests="pytorch_transformer_e0.05_w0.5 pytorch_transformer_e0.1_w0.5 rnnlm_1e_0.45 pytorch_lstm_e0.05_w0.5 pytorch_lstm_e0.1_w0.5"

vote_root_common=${decode_root}/vote_${test_set}_${weight_function}
lcs_root_common=${decode_root}/lcs_${test_set}_${weight_function}

if [ $stage -le 0 ]; then
  echo "$0: doing vote - forward direction"
  python3 local/vote/lm_vote.py --decode_root $decode_root \
          --test_set $test_set \
          --lm_tests $lm_tests \
          --weight_function $weight_function \
          --direction forward \
          vote
fi

if [ $stage -le 1 ]; then
  local/vote/lm_vote_score.sh --lm_tests "$lm_tests vote" ${vote_root_common}_forward
fi

if [ $stage -le 2 ]; then
  echo "$0: doing vote - backward direction"
  python3 local/vote/lm_vote.py --decode_root $decode_root \
          --test_set $test_set \
          --lm_tests $lm_tests \
          --weight_function $weight_function \
          --direction backward \
          vote
fi

if [ $stage -le 3 ]; then
  local/vote/lm_vote_score.sh --lm_tests "$lm_tests vote" ${vote_root_common}_backward
fi

if [ $stage -le 4 ]; then
  echo "$0: doing combine - combine result of forward and backward direction"
  python3 local/vote/lm_vote.py --decode_root $decode_root \
          --test_set $test_set \
          --lm_tests $lm_tests \
          --weight_function $weight_function \
          vote-combine
fi

if [ $stage -le 5 ]; then
  local/vote/lm_vote_score.sh --lm_tests "vote" ${vote_root_common}_combine
fi

if [ $stage -le 6 ]; then
  echo "$0: doing lcs - forward direction"
  python3 local/vote/lm_vote.py --decode_root $decode_root \
          --test_set $test_set \
          --lm_tests $lm_tests \
          --weight_function $weight_function \
          --direction forward \
          lcs
fi

if [ $stage -le 7 ]; then
  local/vote/lm_vote_score.sh --lm_tests "vote" ${lcs_root_common}_forward
fi

if [ $stage -le 8 ]; then
  echo "$0: doing lcs - backward direction"
  python3 local/vote/lm_vote.py --decode_root $decode_root \
          --test_set $test_set \
          --lm_tests $lm_tests \
          --weight_function $weight_function \
          --direction backward \
          lcs
fi

if [ $stage -le 9 ]; then
  local/vote/lm_vote_score.sh --lm_tests "vote" ${lcs_root_common}_backward
fi

if [ $stage -le 10 ]; then
  echo "$0: doing combine - combine result of forward and backward direction"
  python3 local/vote/lm_vote.py --decode_root $decode_root \
          --test_set $test_set \
          --lm_tests $lm_tests \
          --weight_function $weight_function \
          lcs-combine
fi

if [ $stage -le 11 ]; then
  local/vote/lm_vote_score.sh --lm_tests "vote" ${lcs_root_common}_combine
fi

if [ $stage -le 12 ]; then
  echo "Vote results:"

  for c in test; do
    echo "$c:"
    echo "--- CER ---"
    for x in ${decode_root}/vote_${c}_*; do [ -d $x ] && echo $(grep WER $x/cer_vote) $x ; done 2>/dev/null
    for x in ${decode_root}/lcs_${c}_*; do [ -d $x ] && echo $(grep WER $x/cer_vote) $x; done 2>/dev/null
  done

  echo ""
fi
