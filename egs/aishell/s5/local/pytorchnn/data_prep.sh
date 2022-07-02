#!/usr/bin/env bash

# This script prepares the data directory for PyTorch based neural LM training.
# It prepares the following files in a output directory:
# 1. Vocabulary: $dir/words.txt copied from data/lang/words.txt.
# 2. Training and test data: $dir/{train/valid/test}.txt with each sentence per line.


# Begin configuration section.
stage=0
train=data/local/train/text
valid=data/local/dev/text
test=data/local/test/text

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

set -e

if [ $# != 1 ]; then
   echo "Usage: $0 <dest-dir>"
   echo "For details of what the script does, see top of script file"
   exit 1;
fi

dir=$1 # data/pytorchnn/
mkdir -p $dir

for f in $train $valid $test; do
    [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1
done

cp $train $dir/train.org.txt
cp $valid $dir/valid.org.txt
cp $test $dir/test.org.txt
for data in train valid test; do
  cat $dir/${data}.org.txt | cut -d ' ' -f2- | tr 'A-Z' 'a-z' > $dir/$data.txt
  rm $dir/${data}.org.txt
done

mkdir -p $dir/config

# Symbol for unknown words
echo "<SPOKEN_NOISE>" >$dir/config/oov.txt
cp data/lang/words.txt $dir/
# Make sure words.txt contains the symbol for unknown words
if ! grep -w '<SPOKEN_NOISE>' $dir/words.txt >/dev/null; then
  n=$(cat $dir/words.txt | wc -l)
  echo "<SPOKEN_NOISE> $n" >> $dir/words.txt
fi

echo "Data preparation done."
