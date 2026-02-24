#!/bin/bash
#$ -N train_fbank_shallow
#$ -q long.q@supergpu*
#$ -l ram_free=64G,mem_free=64G
#$ -l matylda6=0.5,scratch=0.5
#$ -l gpu=4,gpu_ram=16G
#$ -o /mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/job_logs/train_fbank_shallow.o
#$ -e /mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/job_logs/train_fbank_shallow.e
N_GPUS=4
EXPERIMENT="train_fbank_shallow"
#
# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger checkpoints
ulimit -f unlimited
ulimit -v unlimited
ulimit -u 4096


WORK_DIR=/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm
cd $WORK_DIR
. path.sh

args=(
    --out_dir "$WORK_DIR/exp/$EXPERIMENT"
    --speech_enc_id "openai/whisper-small.en"
    --llm_id "allenai/OLMo-2-0425-1B-Instruct"

    --lr 1e-4
    --wdecay 0.0005
    --steps 30000
    --eval_steps 500
    --warmup 2000
    --nj 2 # 3
    --shuffle
    --bsize 6 # per device
    --gradient_accumulation_steps 2
    --seed 42
    --logging_steps 10
    #--limit_eval_steps 10

    --train_split train
    --validation_split val
    --test_splits dev5

    --encoder_d_model 768

    --norm_first
    --downsampling_factor 6
    --hidden_size 1024
    --num_heads 16
    --intermediate_size 4096
    --num_layers 2
    
    --do_train
    --do_generate

    #--from_pretrained "/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/train_test"
)

echo "Running with args: ${args[@]}"
export CUDA_VISIBLE_DEVICES=`free-gpus.sh $N_GPUS`
echo $CUDA_VISIBLE_DEVICES
if [ "$N_GPUS" -gt 1 ]; then
    torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainer_fbank.py "${args[@]}"
else
    python src/trainer_fbank.py "${args[@]}" 
fi
