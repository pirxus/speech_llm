# AGENTS.md - Development Guide for AI Coding Agents

This document provides coding agents with essential information about this speech LLM research codebase.

## Project Overview

**Purpose**: Research codebase for training speech-to-text models using frozen Whisper encoders and instruction-following LLMs (OLMo, LLaMA) connected via a trainable transformer-based connector.

**Tech Stack**: Python, PyTorch, Hugging Face Transformers, Accelerate, SGE cluster

**Structure**:
- `src/` - Core training and model code
- `scripts/` - SGE job submission scripts
- `recipes/` - Dataset preparation scripts
- `exp/` - Experiment outputs (gitignored)

---

## Environment Setup

### Activate Environment
```bash
source path.sh          # offline mode (default)
source path.sh 0        # online mode
```

This activates the conda environment and sets:
- `PYTHONPATH` to include `src/`
- `HF_HOME=/mnt/matylda6/isedlacek/hugging-face`
- `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, `HF_HUB_OFFLINE`

---

## Build/Run/Test Commands

### Training

**Single GPU (Direct)**:
```bash
python src/trainer_base.py --out_dir exp/my_experiment \
    --speech_enc_id openai/whisper-small.en \
    --llm_id allenai/OLMo-2-0425-1B-Instruct \
    --do_train --do_generate
```

**Multi-GPU (Distributed)**:
```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 src/trainer_base.py [args...]
```

**Cluster Submission**:
```bash
qsub scripts/train.sh              # Main training job
qsub scripts/train_pre.sh          # Pre-norm variant
qsub scripts/train_post_wlg.sh    # WLG variant
```

### Evaluation

**Calculate WER**:
```bash
python src/calculate_wer.py --input exp/my_experiment/test_predictions.json
```

**Analyze Multiple Experiments**:
```bash
python src/calculate_wer.py --input exp/exp1/test_predictions.json exp/exp2/test_predictions.json
```

### Testing

**Note**: No formal test framework (pytest/unittest) exists. Testing is done through:
1. Validation splits during training (`--validation_split val`)
2. Test set generation (`--do_generate --test_splits dev5`)
3. WER calculation on predictions

---

## Code Style & Conventions

### Import Organization
Follow this order (as seen in codebase):
```python
# 1. Standard library
import argparse
import os
import sys
import json

# 2. Third-party core (torch, numpy)
import torch
import torch.nn as nn
import numpy as np

# 3. Third-party ML libraries
from datasets import load_from_disk
from transformers import AutoProcessor, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm

# 4. Local imports (relative or absolute)
from utils import create_logger, get_batch
from speech_llm_model import SpeechLLMBase
from connector import Connector
```

### Naming Conventions

**Variables & Functions**: `snake_case`
```python
speech_processor = AutoProcessor.from_pretrained(args.speech_enc_id)
train_loader = DataLoader(...)
def create_logger(log_file_base: str, verbose: bool):
def stacking_downsampler(embeds, factor=6):
```

**Classes**: `PascalCase`
```python
class SpeechLLMBase(nn.Module):
class DefaultASRCollator:
class SinusoidalPositionalEmbedding(nn.Module):
class TestSetWrapper(torch.utils.data.Dataset):
```

**Constants**: Use descriptive names, sometimes `UPPERCASE`
```python
N_GPUS = 2
EXPERIMENT = "train_post_norm"
```

**Private Methods**: No convention for private methods; use descriptive names

### Type Hints

**Partial adoption**: Use type hints where helpful, especially for function signatures:
```python
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: Optional[int], 
                      decoder_start_token_id: Optional[int]):
    ...

def create_logger(log_file_base: str, verbose: bool):
    ...

def load_text(fname: Union[str, list], subset_ixs=None, ignore_ixs=None, 
              remove_punc=False):
    ...
```

Type hints are not required for all variables or simple helper functions.

### Formatting & Style

**Indentation**: 4 spaces (standard Python)

**Line Length**: No strict limit, but keep reasonable (~100-120 chars)

**String Quotes**: Prefer double quotes `"..."` for strings (consistent in codebase)

**Docstrings**: Use simple descriptions, not always full docstring format:
```python
def shift_tokens_right(input_ids: torch.Tensor, ...):
    """
    Shift input ids one token to the right.
    """
```

**Comments**: Use inline comments for complex logic:
```python
# common for both branches
speech_enc_out = self.speech_encoder(speech_feats, attention_mask=attention_mask).last_hidden_state

# FIXME -- adapt to the new scenario, think about how to store lora adapters
return {"connector": self.connector.state_dict(**kwargs)}
```

---

## Additional Notes

- **No linting config**: Code follows PEP 8 loosely; no black/ruff/flake8 config present
- **No package structure**: Direct script execution, not an installable package
- **Research-focused**: Prioritizes experimentation over production-readiness
- **Cluster-dependent**: Designed for SGE cluster with specific resource requirements
