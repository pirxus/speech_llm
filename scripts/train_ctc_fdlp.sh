#!/bin/bash
#$ -N train_ctc_fdlp
#$ -q long.q@supergpu*
#$ -l ram_free=64G,mem_free=64G
#$ -l matylda6=0.5,scratch=0.5
#$ -l gpu=1,gpu_ram=40G
#$ -o /mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/job_logs/train_ctc_fdlp.o
#$ -e /mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/job_logs/train_ctc_fdlp.e
N_GPUS=1
EXPERIMENT="train_ctc_fdlp"
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
    --data_path "/mnt/scratch/tmp/isedlacek/data/how2"

    --lr 1e-3
    --wdecay 0.0005
    --max_grad_norm 5.0
    --steps 30000
    --eval_steps 500
    --warmup 2000
    --nj 1
    --shuffle
    --bsize 2 # per device
    --gradient_accumulation_steps 1
    --seed 42
    --logging_steps 10
    --n_best 1
    --early_stopping_patience 5
    #--limit_eval_steps 10

    --train_split train
    --validation_split val
    --label_column transcription

    --d_model 512
    --num_encoder_layers 8
    --num_attention_heads 8
    --intermediate_size 2048
    --dropout 0.1
    --max_source_positions 1500

     #FDLP feature extraction parameters
    --n_filters 80
    --coeff_num 80
    --coeff_range "1,80"
    --order 80
    --fduration 1.5
    --frate 100
    --overlap_fraction 0.5
    
    --do_train

    #--from_pretrained "$WORK_DIR/exp/train_ctc_fdlp"
)

echo "Running with args: ${args[@]}"
export CUDA_VISIBLE_DEVICES=`free-gpus.sh $N_GPUS`
echo $CUDA_VISIBLE_DEVICES
if [ "$N_GPUS" -gt 1 ]; then
    torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS src/trainer_ctc_fdlp.py "${args[@]}"
else
    python src/trainer_ctc_fdlp.py "${args[@]}" 
fi
