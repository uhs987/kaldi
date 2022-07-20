#!/usr/bin/env bash

set -e

stage=0
lm_tests=""

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;

# begin configuration section.
cmd=run.pl

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <vote-root>"
  echo " Options:"
  echo "  --lm-tests : test sets to be decoded"

  exit 1;
fi

vote_root=$1

if [ $stage -le 0 ] ; then
  files=($vote_root/scoring_kaldi/test_filt.txt)
  for lm_test in $lm_tests; do
    files+=($vote_root/scoring_kaldi/${lm_test}.txt)
  done

  for f in "${files[@]}" ; do
    fout=${f%.txt}.chars.txt
    if [ -x local/character_tokenizer ]; then
      cat $f |  local/character_tokenizer > $fout
    else
      cat $f |  perl -CSDA -ane '
        {
          print $F[0];
          foreach $s (@F[1..$#F]) {
            if (($s =~ /\[.*\]/) || ($s =~ /\<.*\>/) || ($s =~ "!SIL")) {
              print " $s";
            } else {
              @chars = split "", $s;
              foreach $c (@chars) {
                print " $c";
              }
            }
          }
          print "\n";
        }' > $fout
    fi
  done

  mkdir -p $vote_root/scoring_kaldi/log

  for lm_test in $lm_tests; do
    $cmd $vote_root/scoring_kaldi/log/score.cer.${lm_test}.log \
      cat $vote_root/scoring_kaldi/${lm_test}.chars.txt \| \
      compute-wer --text --mode=present \
      ark:$vote_root/scoring_kaldi/test_filt.chars.txt  ark,p:- ">&" $vote_root/cer_$lm_test || exit 1;
  done
fi
