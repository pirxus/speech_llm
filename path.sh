#!/usr/bin/env bash

# shellcheck source=/dev/null

#source /mnt/matylda4/kesiraju/envs/jret/bin/activate
#source /mnt/matylda6/isedlacek/miniconda3/bin/activate /mnt/matylda6/isedlacek/envs/huggingface_asr
source /mnt/matylda6/isedlacek/miniconda3/bin/activate /mnt/matylda6/isedlacek/miniconda3/envs/gemma

mode=${1:-1}

if [ "${mode}" = "1" ]; then
    echo "Setting to OFFLINE mode. Pass 0 as argument to change to ONLINE mode."
elif [ "${mode}" = "0" ]; then
    echo "Setting to ONLINE mode. Pass 1 as argument or nothing to change to OFFLINE mode."
else
    echo "0 or 1 no other option allowed"
    exit 0;
fi

#export PYTHONPATH=/mnt/matylda4/kesiraju/code/pylibs/:${PYTHONPATH}
#export PYTHONPATH=/mnt/matylda6/isedlacek/projects/joint_retriever/src/:${PYTHONPATH}
export PYTHONPATH=/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/src/:${PYTHONPATH}
#export PYTHONPATH=/mnt/matylda4/kesiraju/code/SONAR/:${PYTHONPATH}
#export TORCH_HOME="/mnt/matylda4/kesiraju/.torch_models/"
#export TORCH_HUB="/mnt/matylda4/kesiraju/.torch_models/"
#export SENTENCE_TRANSFORMERS_HOME="/mnt/matylda4/kesiraju/code/sentence-transformers/cache/"
export TRANSFORMERS_OFFLINE=${mode}
export HF_DATASETS_OFFLINE=${mode}
export HF_HUB_OFFLINE=${mode}

export HF_HOME=/mnt/matylda6/isedlacek/hugging-face
# export TRANSFORMERS_CACHE=/mnt/matylda4/kesiraju/hugging-face/
#export HF_DATASETS_CACHE=$HF_HOME/datasets
#export HF_MODULES_CACHE=$HF_HOME/modules
#export HF_METRICS_CACHE=/mnt/matylda4/kesiraju/hugging-face/metrics
