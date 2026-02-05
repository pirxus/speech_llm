import torch
import random
from contextlib import contextmanager
from transformers import WhisperFeatureExtractor

@contextmanager
def left_padding(tokenizer):
    original_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = 'left'
        yield
    finally:
        tokenizer.padding_side = original_side

@contextmanager
def right_padding(tokenizer):
    original_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = 'right'
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
        prompt_prefix="Transcribe the following speech: ",
        prompt_suffix="\nTranscript:",
        label_column='labels',
        audio_column='audio',
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


        if self.prompt_prefix not in [None, '']:
            with nadd_eos(self.tokenizer):
                prompt_prefix_ids = self.tokenizer(
                    [self.prompt_prefix for _ in batch],
                    return_attention_mask=True,
                    padding="longest",
                    return_tensors="pt",
                )

        else:
            prompt_prefix_ids = None


        if self.prompt_suffix not in [None, '']:
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
                [ item[self.label_column] for item in batch ],
                return_attention_mask=True,
                padding="longest",
                return_tensors="pt",
            )

        labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)

        # Process audio and stack embeddings
        audio_features = self.feature_extractor(
            [audio[self.audio_column]['array'] for audio in batch],
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length" if isinstance(self.feature_extractor, WhisperFeatureExtractor) else "longest",
            return_attention_mask=True,
        )

        input_features = audio_features.input_features
        audio_attention_mask = audio_features.attention_mask

        if isinstance(self.feature_extractor, WhisperFeatureExtractor):
            audio_attention_mask = torch.ones_like(audio_attention_mask)

        if hasattr(batch[0], 'item_idx'):
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "prompt_prefix_ids": prompt_prefix_ids['input_ids'],
                "prompt_prefix_mask": prompt_prefix_ids['attention_mask'],
                "prompt_suffix_ids": prompt_suffix_ids['input_ids'], 
                "prompt_suffix_mask": prompt_suffix_ids['attention_mask'], 
                "labels": labels,
                "item_indices": [ item['item_idx'] for item in batch ],
                "labels_text": [ item[self.label_column] for item in batch ],
            }
        else:
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "prompt_prefix_ids": prompt_prefix_ids['input_ids'],
                "prompt_prefix_mask": prompt_prefix_ids['attention_mask'],
                "prompt_suffix_ids": prompt_suffix_ids['input_ids'], 
                "prompt_suffix_mask": prompt_suffix_ids['attention_mask'], 
                "labels": labels,
            }
