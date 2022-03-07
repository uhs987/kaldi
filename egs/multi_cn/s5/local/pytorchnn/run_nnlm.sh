#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2021  Ke Li

# This script trains an RNN (LSTM and GRU) or Transformer-based
# language model with PyTorch and performs N-best and lattice rescoring.
# More details about the lattice rescoring can be found in the paper:
# https://www.danielpovey.com/files/2021_icassp_parallel_lattice_rescoring.pdf
# The N-best rescoring is in a batch computation mode as well. It is thus much
# faster than N-best rescoring in local/rnnlm/run_tdnn_lstm.sh.


# Begin configuration section.
stage=0
ac_model_dir=exp/chain_cleaned/tdnn_cnn_1a_sp
decode_dir_suffix=pytorch_transformer
pytorch_path=exp/pytorch_transformer
nn_model=$pytorch_path/model.pt

model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=512 # 650 for LSTM (for reproducing perplexities and WERs above)
hidden_dim=512 # 650 for LSTM
nlayers=6 # 2 for LSTM
nhead=8
learning_rate=0.1 # 5 for LSTM
seq_len=100
dropout=0.1

lm_type=
test_sets=""
epsilon=0.5

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$decode_cmd

set -e

if [ $# != 1 ]; then
  echo "Usage: $0 [options] <lm-name>"
  echo " Options:"
  echo "  --lm-type [origin|nointerp|interp] : type of LM to use for decode."
  echo "  --model-type [LSTM|Transformer]"
  echo "  --epsilon : epsilon for lattice rescore"

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
    echo "Invalid --lm-type option: $lm_type"
    exit 1
    ;;
esac

case $model_type in
  LSTM)
    embedding_dim=650
    hidden_dim=650
    nlayers=2
    learning_rate=5

    decode_dir_suffix=pytorch_lstm
    pytorch_path=exp/pytorch_lstm
    ;;
  GRU)
    embedding_dim=650
    hidden_dim=650
    nlayers=2
    learning_rate=5

    decode_dir_suffix=pytorch_gru
    pytorch_path=exp/pytorch_gru
    ;;
  Transformer)
    ;;
  *)
    echo "Invalid --model-type option: $model_type"
    exit 1
    ;;
esac

data_dir=data/pytorchnn/${lm_name}${lm_suffix}
pytorch_path=$pytorch_path/${lm_name}${lm_suffix}
nn_model=$pytorch_path/model.pt

# Check if PyTorch is installed to use with python
if python3 steps/pytorchnn/check_py.py 2>/dev/null; then
  echo PyTorch is ready to use on the python side. This is good.
else
  echo PyTorch not found on the python side.
  echo Please install PyTorch first. For example, you can install it with conda:
  echo "conda install pytorch torchvision cudatoolkit=10.2 -c pytorch", or
  echo with pip: "pip install torch torchvision". If you already have PyTorch
  echo installed somewhere else, you need to add it to your PATH.
  echo Note: you need to install higher version than PyTorch 1.1 to train Transformer models
  exit 1
fi

if [ $stage -le 0 ]; then
  text=data/train_combined/text
  lm_text=data/local/lm/text.$lm_name
  if $lm_lexicon; then
    words=data/lang/${lm_name}${lm_suffix}/words.txt
  else
    words=data/lang/words.txt
  fi

  mkdir -p $data_dir

  echo -n >$data_dir/valid.txt
  # hold out one in every 50 lines as dev data.
  if $lm_text_only; then
    cat $lm_text | cut -d ' ' -f2- | awk -v data_dir=$data_dir '{if(NR%50 == 0) { print >data_dir"/valid.txt"; } else {print;}}' >$data_dir/lm.txt

    mv $data_dir/lm.txt $data_dir/train.txt
  else
    cat $text | cut -d ' ' -f2- | awk -v data_dir=$data_dir '{if(NR%50 == 0) { print >data_dir"/valid.txt"; } else {print;}}' >$data_dir/multi_cn.txt

    cat $lm_text | cut -d ' ' -f2- | awk '{print;}' >$data_dir/lm.txt

    # Merge training data of SWBD and Fisher (ratio 3:1 to match Kaldi RNNLM's preprocessing)
    cat $data_dir/multi_cn.txt $data_dir/lm.txt $data_dir/multi_cn.txt $data_dir/multi_cn.txt > $data_dir/train.txt
  fi

  echo -n >$data_dir/test.txt
  for decode_set in $test_sets; do
    cat data/$decode_set/test/text | cut -d ' ' -f2- | awk '{print;}' >$data_dir/test.txt
  done

  mkdir -p $data_dir/config

  # Symbol for unknown words
  echo "<UNK>" >$data_dir/config/oov.txt
  cp $words $data_dir/
  # Make sure words.txt contains the symbol for unknown words
  if ! grep -w '<UNK>' $data_dir/words.txt >/dev/null; then
    n=$(cat $data_dir/words.txt | wc -l)
    echo "<UNK> $n" >> $data_dir/words.txt
  fi
fi

if [ $stage -le 1 ]; then
  # Train a PyTorch neural network language model.
  echo "Start neural network language model training."
  START=$SECONDS
  $cuda_cmd $pytorch_path/log/train.log utils/parallel/limit_num_gpus.sh \
    python3 steps/pytorchnn/train.py --data $data_dir \
            --model $model_type \
            --emsize $embedding_dim \
            --nhid $hidden_dim \
            --nlayers $nlayers \
            --nhead $nhead \
            --lr $learning_rate \
            --dropout $dropout \
            --seq_len $seq_len \
            --oov "\<UNK\>" \
            --clip 1.0 \
            --batch-size 32 \
            --epoch 64 \
            --save $nn_model \
            --tied \
            --cuda
  END=$SECONDS
  DURATION=$((END-START))
  echo "The script runs for $DURATION seconds"
fi

if [ $stage -le 2 ]; then
  echo "$0: Perform N-best rescoring on $ac_model_dir with a $model_type LM."
  START=$SECONDS
  for decode_set in $test_sets; do
      decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
      steps/pytorchnn/lmrescore_nbest_pytorchnn.sh \
        --cmd "$cmd --mem 4G" \
        --N 20 \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight 0.8 \
        --oov_symbol "'<UNK>'" \
        data/lang_$LM $nn_model $data_dir/words.txt \
        data/${decode_set}/test_hires ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_nbest
  done
  END=$SECONDS
  DURATION=$((END-START))
  echo "The script runs for $DURATION seconds"
fi

if [ $stage -le 3 ]; then
  echo "$0: Perform lattice rescoring on $ac_model_dir with a $model_type LM."
  #START=$SECONDS
  for decode_set in $test_sets; do
    for n in 1 2 3 4 5 6 7 8 9 10; do
      decode_dir=${ac_model_dir}/decode_${decode_set}-${n}_${LM}
      steps/pytorchnn/lmrescore_lattice_pytorchnn.sh \
        --cmd "$cmd --mem 4G" \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight 0.8 \
        --beam 5 \
        --epsilon $epsilon \
        --oov_symbol "'<UNK>'" \
        data/lang_$LM $nn_model $data_dir/words.txt \
        data/${decode_set}/test_hires-$n ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_e$epsilon
    done
  done
  #END=$SECONDS
  #DURATION=$((END-START))
  #echo "The script runs for $DURATION seconds"
fi

if [ $stage -le 4 ]; then
  echo "$lm_name results:"

  for c in $test_sets; do
    echo "$c:"
    echo "--- CER ---"
    for x in exp/*/*/decode_${c}_${LM}_${decode_dir_suffix}_nbest; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    for n in 1 2 3 4 5 6 7 8 9 10; do
      for x in exp/*/*/decode_${c}-${n}_${LM}_${decode_dir_suffix}*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    done
    echo "--- WER ---"
    for x in exp/*/*/decode_${c}_${LM}_${decode_dir_suffix}_nbest; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
    for n in 1 2 3 4 5 6 7 8 9 10; do
      for x in exp/*/*/decode_${c}-${n}_${LM}_${decode_dir_suffix}*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
    done
  done

  echo ""
fi

exit 0
