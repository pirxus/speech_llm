import os
import torch
import re
import sys
import platform
import string
from glob import glob
import json
import logging
from typing import Union
import numpy as np
import subprocess


logger = logging.getLogger(__name__)


class TestSetWrapper(torch.Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['item_idx'] = idx
        return item


def create_logger(log_file_base: str, verbose: bool):
    """Create logger"""

    os.makedirs(os.path.dirname(log_file_base), exist_ok=True)
    # ,%(msecs)03d

    if os.path.exists(log_file_base + ".log"):
        num = glob(log_file_base + "*.log")
        os.rename(log_file_base + ".log", f"{log_file_base}.{len(num)}.log")

    logging.basicConfig(
        format="%(levelname)-8s - %(asctime)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        filename=f"{log_file_base}.log",
        level=logging.INFO,
        filemode="w",
    )
    print(f"Log file {log_file_base}.log")
    logger = logging.getLogger()
    stdout = logging.StreamHandler(stream=sys.stdout)
    if verbose:
        stdout.setLevel(logging.INFO)
    else:
        stdout.setLevel(logging.WARNING)
    logger.addHandler(stdout)
    # logger.addHandler(RichHandler())

    logger.info(" ".join(sys.argv))
    logger.info("%s", platform.node())
    logger.info(
        "CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "Not_set")
    )

    return logger


def save_chunks(chunks: list, chunk_type: str, out_fname: str):
    """Save chunks to disk"""

    if chunk_type == "embs":
        embs = np.concatenate(chunks)
        np.save(out_fname, embs)
    elif chunk_type == "text":
        with open(out_fname, "w", encoding="utf-8") as fpw:
            for line in chunks:
                fpw.write(line.strip().replace("\n", " ") + "\n")
    elif chunk_type == "dict":
        with open(out_fname, "w", encoding="utf-8") as fpw:
            for line in chunks:
                fpw.write(json.dumps(line.strip(), ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unknown chunk type: {chunk_type}")

    logger.info("Saving %d chunks of type %s to %s", len(chunks), chunk_type, out_fname)


def remove_punctuations(line: str):
    """remove punctuations from line"""

    line = line.translate(
        str.maketrans(
            "",
            "",
            string.punctuation + "’“”‘‘[]“" + "\u200c" + "\u200b" + "\u2060" + "—",
        )
    )
    line = re.sub(r"\s+", " ", line)
    return line


def load_text(
    fname: Union[str, list],
    subset_ixs=None,
    ignore_ixs=None,
    remove_punc=False,
):
    """Load text line by line. Optionally, load only a subset of lines"""

    if subset_ixs is not None:
        if isinstance(subset_ixs, np.ndarray):
            subset_ixs = subset_ixs.tolist()
        subset_ixs = set(subset_ixs)

    if ignore_ixs is not None:
        if isinstance(ignore_ixs, np.ndarray):
            ignore_ixs = ignore_ixs.tolist()
        ignore_ixs = set(ignore_ixs)

    fpr = None
    if isinstance(fname, str):
        assert os.path.exists(fname), f"Cannot find the file at {fname}"
        fpr = open(fname, "r", encoding="utf-8")
    elif isinstance(fname, list):
        fpr = fname
    else:
        raise TypeError(f"fname: {fname} should be a file path or list of strings")

    lst = []
    lno = 0
    for line in fpr:
        line = line.strip()
        if line:
            if subset_ixs:
                if lno not in subset_ixs:
                    lno += 1
                    continue

            if ignore_ixs:
                if lno in ignore_ixs:
                    lno += 1
                    continue

            if remove_punc:
                line = remove_punctuations(line)
            lst.append(line)

        lno += 1

    if isinstance(fname, str):
        fpr.close()

    return lst

def stacking_downsampler(embeds, factor=6):
    mod = embeds.shape[-2] % factor
    if mod != 0:
        # append zeros to both the embeddings and the mask if the sequences are not divisible
        # by downsampling_factor
        appendix = torch.zeros((embeds.shape[0], factor - mod,
                                embeds.shape[-1]), device=embeds.device)
        embeds = torch.hstack((embeds, appendix))

    # perform the stacking downsampling
    embeds = embeds.contiguous().view(
        embeds.shape[0],
        embeds.shape[1] // factor,
        embeds.shape[2] * factor
    )
    
    return embeds

def get_duration(audio_fpath) -> float:
    """Get duration in seconds"""
    cmd = [
        "ffprobe",
        "-show_entries",
        "format=duration",
        "-loglevel",
        "quiet",
        "-of",
        "csv=p=0",
        audio_fpath,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    return float(result.stdout.decode("utf-8"))

def get_batch(train_iter, train_loader, accelerator, logger):
    while True:
        try:
            try:
                batch_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch_dict = next(train_iter)
        except:
            if accelerator.is_local_main_process:
                logger.info("Encountered type error in batch, skipping...")
            continue
        break
    return batch_dict
