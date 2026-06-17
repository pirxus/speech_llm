# %%
import os
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import json
import numpy as np
from tqdm import tqdm
import torchaudio
import re


# %%
class SpanishQADataset(Dataset):
    def __init__(
        self,
        split='train',
        data_dir='/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/recipes/mlc/data',
        retrieval_json=None,
        rephrased_extractive=None,
        chunk_length=6,
        overlap=0,
        topk=10,
        use_impossible=True,
        token_budget=300,
        tokenizer=AutoTokenizer.from_pretrained('BSC-LT/salamandra-7b-instruct'),
        token_overlap=100,
        use_suffix_overlap=False,
    ):
        self.data_dir = data_dir
        self.chunk_length = chunk_length
        self.overlap = overlap
        self.topk = topk
        self.token_budget = token_budget
        self.tokenizer = tokenizer
        self.token_overlap = token_overlap
        self.use_suffix_overlap = use_suffix_overlap

        if split in ['train', 'dev']:
            with open(os.path.join(self.data_dir, f'conversations_{split}.json'), 'r') as f:
                self.conversations_raw = json.load(f)['Spanish']

            # format the conversations so that it does not have to be done on the fly
            self.conversations = {}
            for conv_id, conversation in self.conversations_raw.items():
                self.conversations[conv_id] = self.stringify_conversation(conversation)

            # load the tts data
            with open(os.path.join(self.data_dir, f'processed_{split}_tts.json'), 'r') as f:
                self.tts_data = json.load(f)

            # load the real data
            with open(os.path.join(self.data_dir, f'processed_{split}_real.json'), 'r') as f:
                self.real_data = json.load(f)

            # combine the tts and real data
            self.data = self.tts_data + self.real_data

        elif split == 'test':
            with open(os.path.join(self.data_dir, f'conversations_{split}.json'), 'r') as f:
                self.conversations_raw = json.load(f)['Spanish']

            # format the conversations so that it does not have to be done on the fly
            self.conversations = {}
            for conv_id, conversation in self.conversations_raw.items():
                self.conversations[conv_id] = self.stringify_conversation(conversation, is_test=True)

            # load the real data
            with open(os.path.join(self.data_dir, f'processed_{split}_real.json'), 'r') as f:
                self.real_data = json.load(f)

            self.data = self.real_data

        else:
            raise ValueError(f"Invalid split '{split}' specified")

        self.conversations_chunked = {
            conv_id: self.chunk_conversation(conversation, chunk_length=self.chunk_length, overlap=self.overlap, token_budget=self.token_budget, token_overlap=self.token_overlap, use_suffix_overlap=self.use_suffix_overlap) for conv_id, conversation in self.conversations.items()
        }

        # load the rephrased extractive answers -- use them to replace the extractive answers and store the original extractive answer in a different key
        if rephrased_extractive is not None:
            with open(rephrased_extractive, 'r') as f:
                rephrased = json.load(f)

            for item in tqdm(self.data, desc="Replacing extractive answers with natural sentences..."):
                if item['type'] == 'extractive':
                    item['answer_extractive'] = item['answer']
                    item['answer'] = rephrased[item['uuid']]['rephrased_answer']


        # finally, load the retrieval json
        if retrieval_json is not None:
            with open(retrieval_json, 'r') as f:
                self.retrieval = json.load(f)
        else:
            self.retrieval = None

        # discard all impossible questions if specified
        if not use_impossible:
            filtered_data = []
            for item in self.data:
                if item['type'] != 'impossible':
                    filtered_data.append(item)

            self.data = filtered_data


    def chunk_conversation(self, conversation, chunk_length=10, overlap=2, token_budget=None, token_overlap=0, use_suffix_overlap=False):
        if token_budget is not None:
            return self._chunk_by_token_budget(conversation, token_budget=token_budget, token_overlap=token_overlap, use_suffix_overlap=use_suffix_overlap)

        # Original turn-based chunking: split into windows of chunk_length turns with overlap
        chunks = []
        for i in range(0, len(conversation.splitlines()), chunk_length - overlap):
            chunk = conversation.splitlines()[i:i+chunk_length]
            chunks.append(chunk)

        if len(chunks[-1]) <= overlap: chunks = chunks[:-1]

        return [ '\n'.join(chunk) for chunk in chunks ]

    def _chunk_by_token_budget(self, conversation, token_budget=500, token_overlap=0, use_suffix_overlap=False):
        """
        Chunk a conversation into pieces of at most `token_budget` tokens, with
        optional token-counted context windows on either side.

        Strategy:
          Pass 1 — build core chunks: greedily accumulate segments up to
            `token_budget` tokens, splitting long individual turns at sentence
            boundaries.
          Pass 2 — attach overlaps: prepend up to `token_overlap` tokens taken
            from the end of the previous core chunk (prefix overlap), and
            optionally append up to `token_overlap` tokens taken from the start
            of the next core chunk (suffix overlap, enabled by `use_suffix_overlap`).
            Segments are added whole; the loop stops before the first segment that
            would push the overlap window over `token_overlap`.
        """
        def count_tokens(text: str) -> int:
            if self.tokenizer is not None:
                return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
            # Heuristic fallback: ~1.3 sub-word tokens per whitespace token (Spanish)
            return max(1, int(len(text.split()) * 1.3))

        def split_long_turn(line: str) -> list:
            """Split a single turn at sentence boundaries if it exceeds the budget."""
            if count_tokens(line) <= token_budget:
                return [line]

            # Separate "Turno N, Hablante M: " prefix from the spoken text
            if ': ' in line:
                prefix, text = line.split(': ', 1)
                prefix = prefix + ': '
                continued_prefix = prefix[:-2] + ' (continuación): '
            else:
                prefix, text = '', line
                continued_prefix = '(continuación): '

            sentences = re.split(r'(?<=[.!?]) +', text)

            sub_segments = []
            current_sentences = []
            prefix_tokens = count_tokens(prefix)
            current_tokens = prefix_tokens

            for sent in sentences:
                t = count_tokens(sent)
                if current_tokens + t > token_budget and current_sentences:
                    active_prefix = prefix if not sub_segments else continued_prefix
                    sub_segments.append(active_prefix + ' '.join(current_sentences))
                    current_sentences = [sent]
                    current_tokens = count_tokens(continued_prefix) + t
                else:
                    current_sentences.append(sent)
                    current_tokens += t

            if current_sentences:
                active_prefix = prefix if not sub_segments else continued_prefix
                sub_segments.append(active_prefix + ' '.join(current_sentences))

            return sub_segments

        segments = []
        for line in conversation.splitlines():
            if line.strip():
                segments.extend(split_long_turn(line))

        # Pass 1: build core chunks, no overlap
        core_chunks = []
        current_chunk = []
        current_tokens = 0

        for seg in segments:
            seg_tokens = count_tokens(seg)
            if current_tokens + seg_tokens > token_budget and current_chunk:
                core_chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(seg)
            current_tokens += seg_tokens

        if current_chunk:
            core_chunks.append(current_chunk)

        def truncate_to_tail(seg: str, budget: int) -> str:
            """Return the last ~budget tokens of seg, truncating at sentence boundaries.
            Prepends '[...]' when the segment had to be cut."""
            if count_tokens(seg) <= budget:
                return seg
            if ': ' in seg:
                prefix, text = seg.split(': ', 1)
                prefix = prefix + ': '
            else:
                prefix, text = '', seg
            prefix_tokens = count_tokens(prefix)
            sentences = re.split(r'(?<=[.!?]) +', text)
            kept = []
            tokens = prefix_tokens
            for sent in reversed(sentences):
                t = count_tokens(sent)
                if tokens + t > budget and kept:
                    break
                kept.insert(0, sent)
                tokens += t
            truncated = len(kept) < len(sentences)
            return prefix + ('[...] ' if truncated else '') + ' '.join(kept)

        def truncate_to_head(seg: str, budget: int) -> str:
            """Return the first ~budget tokens of seg, truncating at sentence boundaries.
            Appends '[...]' when the segment had to be cut."""
            if count_tokens(seg) <= budget:
                return seg
            if ': ' in seg:
                prefix, text = seg.split(': ', 1)
                prefix = prefix + ': '
            else:
                prefix, text = '', seg
            prefix_tokens = count_tokens(prefix)
            sentences = re.split(r'(?<=[.!?]) +', text)
            kept = []
            tokens = prefix_tokens
            for sent in sentences:
                t = count_tokens(sent)
                if tokens + t > budget and kept:
                    break
                kept.append(sent)
                tokens += t
            truncated = len(kept) < len(sentences)
            return prefix + ' '.join(kept) + (' [...]' if truncated else '')

        # Pass 2: attach token-limited prefix / suffix overlap windows.
        # Greedy whole-segment pass first; if nothing fits (boundary segment too
        # large), fall back to sentence-level truncation of that one segment.
        result = []
        for i, chunk_segs in enumerate(core_chunks):
            prefix_segs = []
            if token_overlap > 0 and i > 0:
                tokens = 0
                for seg in reversed(core_chunks[i - 1]):
                    t = count_tokens(seg)
                    if tokens + t > token_overlap:
                        break
                    prefix_segs.insert(0, seg)
                    tokens += t
                if not prefix_segs:
                    prefix_segs = [truncate_to_tail(core_chunks[i - 1][-1], token_overlap)]

            suffix_segs = []
            if use_suffix_overlap and token_overlap > 0 and i < len(core_chunks) - 1:
                tokens = 0
                for seg in core_chunks[i + 1]:
                    t = count_tokens(seg)
                    if tokens + t > token_overlap:
                        break
                    suffix_segs.append(seg)
                    tokens += t
                if not suffix_segs:
                    suffix_segs = [truncate_to_head(core_chunks[i + 1][0], token_overlap)]

            result.append('\n'.join(prefix_segs + chunk_segs + suffix_segs))

        #for chunk in result:
        #    print(f"Chunk tokens: {count_tokens(chunk)} | Content:\n{chunk}\n{'-'*40}")
        return result

    def stringify_conversation(self, conversation, split='Spanish', add_turn=True, is_test=False):
        SPLITS = ['English/American', 'Spanish']
        assert split in SPLITS, "Invalid split specified"
        if split == "English/American":
            turn_string = "Turn"
            speaker_string = "Speaker"
        else:
            turn_string = "Turno"
            speaker_string = "Hablante"

        if not is_test:
            speaker = 1
            dialogue = []
            for item in conversation:
                if add_turn:
                    dialogue.append(f"{turn_string} {int(item['turn']) + 1}, {speaker_string} {2 if speaker == 0 else speaker}: {' '.join(item['transcript'])}")
                else:
                    dialogue.append(f"{speaker_string} {2 if speaker == 0 else speaker}: {' '.join(item['transcript'])}")

                speaker = int(not speaker)
        else:
            # there are multiple speakers possibly in the test set, so we will just create a translation dict for the speakers
            speakers = []
            for item in conversation:
                if item['speaker'] not in speakers:
                    speakers.append(item['speaker'])

            speaker_dict = {speaker: f"{speaker_string} {i+1}" for i, speaker in enumerate(speakers)}

            dialogue = []
            for item in conversation:
                if add_turn:
                    dialogue.append(f"{turn_string} {int(item['turn']) + 1}, {speaker_dict[item['speaker']]}: {' '.join(item['transcript'])}")
                else:
                    dialogue.append(f"{speaker_dict[item['speaker']]}: {' '.join(item['transcript'])}")

        return '\n'.join(dialogue)

    def __len__(self):
        return len(self.data)

    def load_and_resample_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio, sr, 16000).squeeze().numpy()
        return {
            'array': audio,
            'path': audio_path,
            'sampling_rate': 16000,
        }

    def __getitem__(self, idx):
        qa = self.data[idx]
        conversation = self.conversations[qa['conv_id']]
        chunks = self.conversations_chunked[qa['conv_id']]

        # load the audio
        audio = self.load_and_resample_audio(qa['audio_path'])

        return_dict = {
            'conversation': conversation,
            'chunks': chunks,
            'audio': audio,
            **qa,
        }

        if self.retrieval is not None:
            retrieval = self.retrieval[qa['uuid']]
            topk = retrieval['topk'][:self.topk]

            return_dict.update({
                'topk': topk,
                'topk_chunks': np.array(chunks)[topk],
            })

        return return_dict


# %%

# let's test the dataset
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('BSC-LT/salamandra-7b-instruct')
    """
    dataset = {
            'dev': SpanishQADataset(
                split='dev',
                retrieval_json='/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/m3_text_retrieval.json',
                rephrased_extractive='/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/rephrased_answers.json',
            ),
            'test': SpanishQADataset(
                split='test',
                retrieval_json='/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/m3_text_retrieval.json',
                rephrased_extractive='/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/rephrased_answers.json',
            ),
            'train': SpanishQADataset(
                split='train',
                retrieval_json='/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/m3_text_retrieval.json',
                rephrased_extractive='/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/rephrased_answers.json',
            ),
    }


    def plot_chunk_lengths(dataset: dict[str, SpanishQADataset], tokenizer) -> None:
        import matplotlib.pyplot as plt

        all_lengths = []
        split_lengths = {}

        for split, d in dataset.items():
            lengths = []
            for chunks in d.conversations_chunked.values():
                for chunk in chunks:
                    lengths.append(len(tokenizer(chunk)["input_ids"]))

            split_lengths[split] = lengths
            all_lengths.extend(lengths)

            print(
                f"{split}: n_chunks={len(lengths)}, "
                f"mean={np.mean(lengths):.2f}, median={np.median(lengths):.2f}, max={np.max(lengths)}"
            )

        plt.figure(figsize=(10, 6))
        bins = 50
        for split, lengths in split_lengths.items():
            plt.hist(lengths, bins=bins, alpha=0.4, label=split)

        plt.title("Tokenized Chunk Length Distribution by Split")
        plt.xlabel("Number of tokens")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        #plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(all_lengths, bins=50)
        plt.title("Tokenized Chunk Length Distribution (All Splits)")
        plt.xlabel("Number of tokens")
        plt.ylabel("Frequency")
        plt.tight_layout()
        #plt.show()

    plot_chunk_lengths(dataset, tokenizer)


    # Print a few of the largest chunks in the train split
    train_chunk_entries = []
    long_chunk_counts = {}

    for conv_id, chunks in dataset['train'].conversations_chunked.items():
        for chunk_idx, chunk in enumerate(chunks):
            n_tokens = len(tokenizer(chunk)["input_ids"])
            train_chunk_entries.append((n_tokens, conv_id, chunk_idx, chunk))

            if n_tokens > 1000:
                long_chunk_counts[conv_id] = long_chunk_counts.get(conv_id, 0) + 1

    train_chunk_entries.sort(key=lambda x: x[0], reverse=True)

    print("\nLargest chunks in train split:")
    for rank, (n_tokens, conv_id, chunk_idx, chunk) in enumerate(train_chunk_entries[:5], start=1):
        print(f"\n#{rank} | conv_id={conv_id} | chunk_idx={chunk_idx} | tokens={n_tokens}")
        print(chunk)
        print("-" * 80)

    total_long_chunks = sum(long_chunk_counts.values())
    print(f"\nNumber of train chunks longer than 1000 tokens: {total_long_chunks}")
    print("conv_id -> count of chunks longer than 1000 tokens:")
    if long_chunk_counts:
        for conv_id, count in sorted(long_chunk_counts.items(), key=lambda x: x[0]):
            print(f"{conv_id} -> {count}")
    else:
        print("None")

    print("-" * 80)
    print(f"Total number of conversations in train split: {len(dataset['train'].conversations)}")
    """



    #print(item['topk'])
    #print(len(tokenizer('\n'.join(item['topk_chunks']))['input_ids']))
    #dataset = SpanishQADataset(split='test')
    # print(dataset.conversations['conv0'])
    # print(dataset[0])
    # longer = 0
    # for item in dataset:
    #     print(len(tokenizer(item['conversation'])['input_ids']))
    #     if len(tokenizer(item['conversation'])['input_ids']) > 8192:
    #         print(f"Conversation {item['conv_id']} is too long with {len(tokenizer(item['conversation'])['input_ids'])} tokens")
    #         longer += 1
    # print(f"Number of conversations longer than 8192 tokens: {longer}")

    #dataset = SpanishQADataset(split='train')
    #print(len(dataset))
    #longer = 0
    #for item in dataset:
    #    if len(tokenizer(item['conversation'])['input_ids']) > 8192:
    #        print(f"Conversation {item['conv_id']} is too long with {len(tokenizer(item['conversation'])['input_ids'])} tokens")
    #        longer += 1
    #print(f"Number of conversations longer than 8192 tokens: {longer}")

    dev = SpanishQADataset(
        split='dev',
        retrieval_json='/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/m3_text_retrieval.json',
        rephrased_extractive='/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/rephrased_answers.json',
        token_budget=300,
        tokenizer=tokenizer,
        token_overlap=100,
        use_suffix_overlap=False,
    )
