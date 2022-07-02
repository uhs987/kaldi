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
ac_model_dir=exp/chain/tdnn_1a_sp
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

epsilon=0.5
weight=0.8

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$decode_cmd

set -e

if [ $# != 0 ]; then
  echo "Usage: $0 [options]"
  echo " Options:"
  echo "  --model-type [LSTM|Transformer] : model type"
  echo "  --epsilon : epsilon for lattice rescore"
  echo "  --weight  : weight for nbest rescore"

  exit 1;
fi

case $model_type in
  LSTM)
    embedding_dim=650
    hidden_dim=650
    nlayers=2
    learning_rate=5

    decode_dir_suffix=pytorch_lstm
    pytorch_path=exp/pytorch_lstm
    nn_model=$pytorch_path/model.pt
    ;;
  GRU)
    embedding_dim=650
    hidden_dim=650
    nlayers=2
    learning_rate=5

    decode_dir_suffix=pytorch_gru
    pytorch_path=exp/pytorch_gru
    nn_model=$pytorch_path/model.pt
    ;;
  Transformer)
    ;;
  *)
    echo "Invalid model type: $model_type"
    exit 1
    ;;
esac

data_dir=data/pytorchnn

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
  local/pytorchnn/data_prep.sh $data_dir
fi

if [ $stage -le 1 ]; then
  # Train a PyTorch neural network language model.
  echo "Start neural network language model training."
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
            --oov "\<SPOKEN_NOISE\>" \
            --clip 1.0 \
            --batch-size 32 \
            --epoch 64 \
            --save $nn_model \
            --tied \
            --cuda
fi

LM=test
if [ $stage -le 2 ]; then
  echo "$0: Perform N-best rescoring on $ac_model_dir with a $model_type LM."
  for decode_set in test; do
      decode_dir=${ac_model_dir}/decode_${decode_set}
      steps/pytorchnn/lmrescore_nbest_pytorchnn.sh \
        --cmd "$cmd --mem 4G" \
        --N 20 \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight $weight \
        --oov_symbol "'<SPOKEN_NOISE>'" \
        data/lang_$LM $nn_model $data_dir/words.txt \
        data/${decode_set}_hires ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_nbest_w$weight
  done
fi

if [ $stage -le 3 ]; then
  echo "$0: Perform lattice rescoring on $ac_model_dir with a $model_type LM."
  for decode_set in test; do
      decode_dir=${ac_model_dir}/decode_${decode_set}
      steps/pytorchnn/lmrescore_lattice_pytorchnn.sh \
        --cmd "$cmd --mem 4G" \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight $weight \
        --beam 5 \
        --epsilon $epsilon \
        --oov_symbol "'<SPOKEN_NOISE>'" \
        data/lang_$LM $nn_model $data_dir/words.txt \
        data/${decode_set}_hires ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_e${epsilon}_w$weight
  done
fi

if [ $stage -le 4 ]; then
  echo "NNLM decode results:"

  for c in test; do
    echo "$c:"
    echo "--- CER ---"
    for x in ${ac_model_dir}/decode_${c}_${decode_dir_suffix}_nbest_w*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    for x in ${ac_model_dir}/decode_${c}_${decode_dir_suffix}_e*_w*; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
    echo "--- WER ---"
    for x in ${ac_model_dir}/decode_${c}_${decode_dir_suffix}_nbest_w*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
    for x in ${ac_model_dir}/decode_${c}_${decode_dir_suffix}_e*_w*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done 2>/dev/null
  done

  echo ""
fi

exit 0
