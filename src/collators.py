import torch
import random
import sys
import os
from contextlib import contextmanager
from transformers import WhisperFeatureExtractor

# Add fdlp_spectrogram to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fdlp_spectrogram"))
from fdlp import FDLP


@contextmanager
def left_padding(tokenizer):
    original_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        yield
    finally:
        tokenizer.padding_side = original_side


@contextmanager
def right_padding(tokenizer):
    original_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "right"
        yield
    finally:
        tokenizer.padding_side = original_side


@contextmanager
def nadd_eos(tokenizer):
    try:
        original = tokenizer.add_eos_token
        try:
            tokenizer.add_eos_token = False
            yield
        finally:
            tokenizer.add_eos_token = original
    except:
        yield
    finally:
        pass


@contextmanager
def add_eos(tokenizer):
    try:
        original = tokenizer.add_eos_token
        try:
            tokenizer.add_eos_token = True
            yield
        finally:
            tokenizer.add_eos_token = original
    except:
        yield
    finally:
        pass


@contextmanager
def nadd_bos(tokenizer):
    try:
        original = tokenizer.add_bos_token
        try:
            tokenizer.add_bos_token = False
            yield
        finally:
            tokenizer.add_bos_token = original
    except:
        yield
    finally:
        pass


class DefaultASRCollator:
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        prompt_prefix="<|user|>\nTranscribe the following speech:\n",
        prompt_suffix="\n<|assistant|>\nTranscript:",
        label_column="labels",
        audio_column="audio",
        audio_max_length=None,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

        self.label_column = label_column
        self.audio_column = audio_column
        self.audio_max_length = audio_max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch):
        if self.prompt_prefix not in [None, ""]:
            with nadd_eos(self.tokenizer):
                prompt_prefix_ids = self.tokenizer(
                    [self.prompt_prefix for _ in batch],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )

        else:
            prompt_prefix_ids = None

        if self.prompt_suffix not in [None, ""]:
            prompt_suffix_ids = self.tokenizer(
                [self.prompt_suffix for _ in batch],
                return_attention_mask=True,
                padding="longest",
                padding_side="left",
                add_special_tokens=False,
                return_tensors="pt",
            )
        else:
            prompt_suffix_ids = None

        with nadd_bos(self.tokenizer), add_eos(self.tokenizer):
            labels = self.tokenizer(
                [item[self.label_column] for item in batch],
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)

        # Process audio and stack embeddings
        audio_features = self.feature_extractor(
            [audio[self.audio_column]["array"] for audio in batch],
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length"
            if isinstance(self.feature_extractor, WhisperFeatureExtractor)
            else "longest",
            return_attention_mask=True,
        )

        input_features = audio_features.input_features
        audio_attention_mask = audio_features.attention_mask

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            audio_attention_mask = torch.ones_like(audio_attention_mask)

        if "item_idx" in batch[0]:
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "prompt_prefix_ids": prompt_prefix_ids["input_ids"],
                "prompt_prefix_mask": prompt_prefix_ids["attention_mask"],
                "prompt_suffix_ids": prompt_suffix_ids["input_ids"],
                "prompt_suffix_mask": prompt_suffix_ids["attention_mask"],
                "labels": labels,
                "item_indices": [item["item_idx"] for item in batch],
                "labels_text": [item[self.label_column] for item in batch],
            }
        else:
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "prompt_prefix_ids": prompt_prefix_ids["input_ids"],
                "prompt_prefix_mask": prompt_prefix_ids["attention_mask"],
                "prompt_suffix_ids": prompt_suffix_ids["input_ids"],
                "prompt_suffix_mask": prompt_suffix_ids["attention_mask"],
                "labels": labels,
            }


class FBankCollator:
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        prompt_prefix="<|user|>\nTranscribe the following speech:\n",
        prompt_suffix="\n<|assistant|>\nTranscript:",
        label_column="labels",
        audio_column="audio",
        audio_max_length=None,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

        self.label_column = label_column
        self.audio_column = audio_column
        self.audio_max_length = audio_max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch):
        if self.prompt_prefix not in [None, ""]:
            with nadd_eos(self.tokenizer):
                prompt_prefix_ids = self.tokenizer(
                    [self.prompt_prefix for _ in batch],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )

        else:
            prompt_prefix_ids = None

        if self.prompt_suffix not in [None, ""]:
            prompt_suffix_ids = self.tokenizer(
                [self.prompt_suffix for _ in batch],
                return_attention_mask=True,
                padding="longest",
                padding_side="left",
                add_special_tokens=False,
                return_tensors="pt",
            )
        else:
            prompt_suffix_ids = None

        with nadd_bos(self.tokenizer), add_eos(self.tokenizer):
            labels = self.tokenizer(
                [item[self.label_column] for item in batch],
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)

        # Process audio and stack embeddings
        audio_features = self.feature_extractor(
            [audio[self.audio_column]["array"] for audio in batch],
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )

        input_features = audio_features.input_features
        audio_attention_mask = audio_features.attention_mask

        if "item_idx" in batch[0]:
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "prompt_prefix_ids": prompt_prefix_ids["input_ids"],
                "prompt_prefix_mask": prompt_prefix_ids["attention_mask"],
                "prompt_suffix_ids": prompt_suffix_ids["input_ids"],
                "prompt_suffix_mask": prompt_suffix_ids["attention_mask"],
                "labels": labels,
                "item_indices": [item["item_idx"] for item in batch],
                "labels_text": [item[self.label_column] for item in batch],
            }
        else:
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "prompt_prefix_ids": prompt_prefix_ids["input_ids"],
                "prompt_prefix_mask": prompt_prefix_ids["attention_mask"],
                "prompt_suffix_ids": prompt_suffix_ids["input_ids"],
                "prompt_suffix_mask": prompt_suffix_ids["attention_mask"],
                "labels": labels,
            }


class CTCCollator:
    """
    Collator for CTC training with character-level labels.
    Lowercases text and converts to character indices on the fly.
    """

    def __init__(
        self,
        feature_extractor,
        label_column="transcription",
        audio_column="audio",
        vocab=None,
    ):
        self.feature_extractor = feature_extractor
        self.label_column = label_column
        self.audio_column = audio_column

        # Create character vocabulary if not provided
        if vocab is None:
            # CTC blank token at index 0
            # Then space and lowercase English alphabet
            self.vocab = {
                "<blank>": 0,
                " ": 1,
            }
            # Add lowercase letters a-z
            for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
                self.vocab[char] = i + 2
            # Add common punctuation
            for i, char in enumerate("',.?!-"):
                self.vocab[char] = len(self.vocab)
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab)
        self.blank_id = 0

    def text_to_indices(self, text):
        """Convert text to character indices, lowercase and filter unknown chars"""
        text = text.lower()
        indices = []
        for char in text:
            if char in self.vocab:
                indices.append(self.vocab[char])
            # Skip unknown characters
        return indices

    def __call__(self, batch):
        # Process audio features
        audio_features = self.feature_extractor(
            [item[self.audio_column]["array"] for item in batch],
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )

        input_features = audio_features.input_features
        audio_attention_mask = audio_features.attention_mask

        # Convert text labels to character indices
        label_texts = [item[self.label_column] for item in batch]
        label_indices = [self.text_to_indices(text) for text in label_texts]
        label_lengths = torch.tensor(
            [len(indices) for indices in label_indices], dtype=torch.long
        )

        # Pad labels to same length
        max_label_len = max(len(indices) for indices in label_indices)
        padded_labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
        for i, indices in enumerate(label_indices):
            padded_labels[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)

        if "item_idx" in batch[0]:
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "labels": padded_labels,
                "label_lengths": label_lengths,
                "item_indices": [item["item_idx"] for item in batch],
                "labels_text": label_texts,
            }
        else:
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "labels": padded_labels,
                "label_lengths": label_lengths,
            }


class FDLPCTCCollator:
    """
    Collator for CTC training with FDLP features.
    Uses FDLP feature extraction instead of Whisper filterbanks.
    """

    def __init__(
        self,
        fdlp_config=None,
        label_column="transcription",
        audio_column="audio",
        vocab=None,
    ):
        # Initialize FDLP feature extractor
        if fdlp_config is None:
            # Use default FDLP configuration
            self.fdlp = FDLP(
                n_filters=80,
                coeff_num=80,
                coeff_range='1,80',
                order=80,
                fduration=1.5,
                frate=100,
                overlap_fraction=0.5,
                srate=16000,
            )
        else:
            self.fdlp = FDLP(**fdlp_config)
        
        self.label_column = label_column
        self.audio_column = audio_column

        # Create character vocabulary if not provided
        if vocab is None:
            # CTC blank token at index 0
            # Then space and lowercase English alphabet
            self.vocab = {
                "<blank>": 0,
                " ": 1,
            }
            # Add lowercase letters a-z
            for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
                self.vocab[char] = i + 2
            # Add common punctuation
            for i, char in enumerate("',.?!-"):
                self.vocab[char] = len(self.vocab)
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab)
        self.blank_id = 0

    def text_to_indices(self, text):
        """Convert text to character indices, lowercase and filter unknown chars"""
        text = text.lower()
        indices = []
        for char in text:
            if char in self.vocab:
                indices.append(self.vocab[char])
            # Skip unknown characters
        return indices

    def __call__(self, batch):
        # Extract audio arrays and their lengths
        audio_arrays = [item[self.audio_column]["array"] for item in batch]
        audio_lengths = torch.tensor([len(audio) for audio in audio_arrays], dtype=torch.long)
        
        # Pad audio arrays to the same length
        max_audio_len = max(len(audio) for audio in audio_arrays)
        padded_audio = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)
        for i, audio in enumerate(audio_arrays):
            padded_audio[i, :len(audio)] = torch.tensor(audio, dtype=torch.float32)
        
        # Extract FDLP features
        # FDLP expects (batch, signal_length) and returns (batch, time, n_filters)
        fdlp_feats, feat_lengths = self.fdlp.extract_feats(padded_audio.numpy(), audio_lengths.numpy())
        fdlp_feats = torch.from_numpy(fdlp_feats).float()
        feat_lengths = torch.from_numpy(feat_lengths).long()
        
        # Transpose to (batch, n_filters, time) to match expected input format
        input_features = fdlp_feats.permute(0, 2, 1)
        
        # Create attention mask based on feature lengths
        max_feat_len = input_features.shape[2]
        audio_attention_mask = torch.zeros(len(batch), max_feat_len, dtype=torch.long)
        for i, feat_len in enumerate(feat_lengths):
            print(feat_len)
            audio_attention_mask[i, :feat_len] = 1

        # Convert text labels to character indices
        label_texts = [item[self.label_column] for item in batch]
        label_indices = [self.text_to_indices(text) for text in label_texts]
        label_lengths = torch.tensor(
            [len(indices) for indices in label_indices], dtype=torch.long
        )

        # Pad labels to same length
        max_label_len = max(len(indices) for indices in label_indices)
        padded_labels = torch.zeros(len(batch), max_label_len, dtype=torch.long)
        for i, indices in enumerate(label_indices):
            padded_labels[i, : len(indices)] = torch.tensor(indices, dtype=torch.long)

        if "item_idx" in batch[0]:
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "labels": padded_labels,
                "label_lengths": label_lengths,
                "item_indices": [item["item_idx"] for item in batch],
                "labels_text": label_texts,
            }
        else:
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "labels": padded_labels,
                "label_lengths": label_lengths,
            }


def create_collator(name, *args, **kwargs):
    collators = {
        "DefaultASRCollator": DefaultASRCollator,
        "FBankCollator": FBankCollator,
        "CTCCollator": CTCCollator,
        "FDLPCTCCollator": FDLPCTCCollator,
    }

    return collators[name](*args, **kwargs)
