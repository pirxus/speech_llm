N_GPUS=1
EXPERIMENT="test"
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

export CUDA_VISIBLE_DEVICES=`free-gpus.sh $N_GPUS`
echo $CUDA_VISIBLE_DEVICES

cd src/qa

python run_m3_retrieval.py
