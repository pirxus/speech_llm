import torch
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
        prompt_prefix="<|user|>\nTranscribe the following speech:\n",
        prompt_suffix="\n<|assistant|>\nTranscript:",
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

        if 'item_idx' in batch[0]:
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


class SalamandraASRCollator:
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        prompt_prefix="<|user|>\nTranscribe the following speech:\n",
        prompt_suffix="\n<|assistant|>\nTranscript:",
        label_column='transcription',
        audio_column='audio',
        audio_max_length=None,
        use_chat_template=True,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

        self.use_chat_template = use_chat_template
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix

        self.label_column = label_column
        self.audio_column = audio_column
        self.audio_max_length = audio_max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch):

        if self.use_chat_template:

            message = [ { "role": "user", "content": "Transcribe el siguiente audio a texto:" } ]
            prompt = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            # get just the prompt prefix part
            prefix = prompt.rpartition('<|im_end|>')[0]
            suffix = ''.join(prompt.rpartition('<|im_end|>')[1:]) + "Claro, aquí está la transcripción:\n"
            labels = [ item[self.label_column] + "<|im_end|>" for item in batch ]


        else:
            prefix = self.prompt_prefix
            suffix = self.prompt_suffix
            labels = [ item[self.label_column] for item in batch ]

        if prefix not in [None, '']:
            prompt_prefix_ids = self.tokenizer(
                [prefix for _ in batch],
                return_attention_mask=True,
                add_special_tokens=False,
                padding="longest",
                return_tensors="pt",
            )
        else:
            prompt_prefix_ids = None

        if suffix not in [None, '']:
            prompt_suffix_ids = self.tokenizer(
                [suffix for _ in batch],
                return_attention_mask=True,
                padding="longest",
                padding_side="left",
                add_special_tokens=False,
                return_tensors="pt",
            )
        else:
            prompt_suffix_ids = None

        labels = self.tokenizer(
            labels,
            return_attention_mask=True,
            padding="longest",
            add_special_tokens=False,
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

        if 'item_idx' in batch[0]:
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

PROMPT = """You are a helpful assistant answering spoken questions about a conversation between a few speakers.

Given the following transcribed conversation excerpts and the user's query in spoken form, transcribe the query and provide a concise answer to the question in the following JSON format: {"question": %question transcript%, "answer": %answer to the question given the context%}

If the question is impossible to answer given the provided context, output a JSON with only the question transcript: {"question": %question transcript%}

Here is the conversation context:
{context}

Here is the spoken query:
"""

PROMPT_SPANISH = """Eres un asistente útil que responde a preguntas habladas sobre una conversación entre varios interlocutores.

Dados los siguientes extractos transcritos de la conversación y la consulta del usuario en formato oral, transcribe la consulta y proporciona una respuesta concisa a la pregunta en el siguiente formato JSON: {"question": "%transcripción de la pregunta%", "answer": "%respuesta a la pregunta dado el contexto%"}

Si resulta imposible responder a la pregunta con el contexto proporcionado, devuelve un JSON que contenga únicamente la transcripción de la pregunta: {"question": "%transcripción de la pregunta%"}

Este es el contexto de la conversación:
{context}

Esta es la consulta hablada:
"""

class SalamandraQACollator:
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        audio_max_length=None,
        audio_column='audio',
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.audio_max_length = audio_max_length
        self.audio_column = audio_column

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, batch):

        message = [ { "role": "user", "content": PROMPT_SPANISH } ]
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        prefix = prompt.rpartition('<|im_end|>')[0]
        #suffix = ''.join(prompt.rpartition('<|im_end|>')[1:]) + "Sure, here's the JSON:\n"
        suffix = ''.join(prompt.rpartition('<|im_end|>')[1:])
        labels_text = []
        for item in batch:
            if item['type'] == 'impossible':
                labels_text.append('{"question": "' + item['question'] + '"}' + "<|im_end|>")
            else:
                labels_text.append('{"question": "' + item['question'] + '", "answer": "' + item['answer'] +'"}' + "<|im_end|>")

        # assemble the prefixes with the chunks
        prefixes = []
        for item in batch:
            chunks = '\n...\n'.join([ chunk for chunk in sorted(item['topk_chunks'], key=lambda x: int(x.split()[1][:-1])) ])
            print(chunks)
            prefixes.append(prefix.replace('{context}', chunks))

        prompt_prefix_ids = self.tokenizer(
            prefixes,
            return_attention_mask=True,
            add_special_tokens=False,
            padding="longest",
            padding_side="left",
            return_tensors="pt",
        )

        prompt_suffix_ids = self.tokenizer(
            [suffix for _ in batch],
            return_attention_mask=True,
            padding="longest",
            padding_side="left",
            add_special_tokens=False,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            labels_text,
            return_attention_mask=True,
            padding="longest",
            add_special_tokens=False,
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

        if 'item_idx' in batch[0]:
            return {
                "speech_feats": input_features,
                "audio_attention_mask": audio_attention_mask,
                "prompt_prefix_ids": prompt_prefix_ids['input_ids'],
                "prompt_prefix_mask": prompt_prefix_ids['attention_mask'],
                "prompt_suffix_ids": prompt_suffix_ids['input_ids'],
                "prompt_suffix_mask": prompt_suffix_ids['attention_mask'],
                "labels": labels,
                "item_indices": [ item['item_idx'] for item in batch ],
                "labels_text": labels_text,
                "id": [ item['uuid'] for item in batch ],
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

def create_collator(name, *args, **kwargs):
    collators = {
        "DefaultASRCollator": DefaultASRCollator,
        "SalamandraASRCollator": SalamandraASRCollator,
        "SalamandraQACollator": SalamandraQACollator,
    }

    return collators[name](*args, **kwargs)
