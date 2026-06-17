from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import json
import torch
from qa_dataset import SpanishQADataset
import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from collections import defaultdict
from transformers import AutoTokenizer


model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices='cuda')

K = 100
QUESTION_BATCH_SIZE = 32

# TODO: define chunking arguments
dataset_kwargs = dict( 
    tokenizer=AutoTokenizer.from_pretrained('BSC-LT/salamandra-7b-instruct'),
    token_budget=300,
    token_overlap=100,
    use_suffix_overlap=False,
)

data = {}
for split in ['dev', 'test', 'train']:
    dataset = SpanishQADataset(split=split, **dataset_kwargs)
    print(f"Processing split '{split}' with {len(dataset)} items.")

    # Step 1: Group items by conv_id and collect all questions
    conv_id_to_chunks = {}  # conv_id -> chunks list
    conv_id_to_items = defaultdict(list)  # conv_id -> list of items
    all_questions = []
    item_to_idx = {}  # Map item uuid to its index in all_questions
    
    for idx, item in enumerate(dataset):
        conv_id = item['conv_id']
        uuid = item['uuid']
        
        if conv_id not in conv_id_to_chunks:
            conv_id_to_chunks[conv_id] = item['chunks']
        
        conv_id_to_items[conv_id].append(item)
        all_questions.append(item['question'])
        item_to_idx[uuid] = idx

    print(f"Found {len(conv_id_to_chunks)} unique conversations.")

    # Step 2: Encode all chunks per conversation
    conv_id_to_e_chunks = {}
    print("Encoding chunks...")
    for conv_id, chunks in tqdm(conv_id_to_chunks.items(), desc="Encoding chunks"):
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                e_c = model.encode(chunks, batch_size=32, max_length=2048)['dense_vecs']
        conv_id_to_e_chunks[conv_id] = torch.tensor(e_c, dtype=torch.float32)

    # Step 3: Encode all questions in batches
    print("Encoding questions...")
    e_questions = []
    for batch_start in tqdm(range(0, len(all_questions), QUESTION_BATCH_SIZE), desc="Encoding questions"):
        batch_end = min(batch_start + QUESTION_BATCH_SIZE, len(all_questions))
        batch_questions = all_questions[batch_start:batch_end]
        
        with open(os.devnull, "w") as fnull:
            with redirect_stdout(fnull), redirect_stderr(fnull):
                e_qs = model.encode(batch_questions, batch_size=QUESTION_BATCH_SIZE, max_length=2048)['dense_vecs']
        
        e_questions.extend([torch.tensor(e_q, dtype=torch.float32) for e_q in e_qs])

    e_questions = torch.stack(e_questions)  # (num_questions, D)

    # Step 4: Compute retrieval for all items
    print("Computing retrievals...")
    for item in tqdm(dataset, desc="Computing similarities"):
        uuid = item['uuid']
        conv_id = item['conv_id']
        q_idx = item_to_idx[uuid]
        
        e_q = e_questions[q_idx].unsqueeze(0)  # (1, D)
        e_c = conv_id_to_e_chunks[conv_id]  # (N, D)
        
        similarity = e_q @ e_c.T  # (1, N)
        
        topk = min(K, similarity.size(1))
        topk_values, topk_indices = torch.topk(similarity, k=topk, dim=1)
        
        data[uuid] = {
            'topk': topk_indices.squeeze(0).tolist(),
            'similarities': topk_values.squeeze(0).tolist(),
            'question': item['question'],
        }

# Save the data
output_dir = os.path.dirname('/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/m3_text_retrieval.json')
os.makedirs(output_dir, exist_ok=True)
with open('/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/m3_text_retrieval.json', 'w') as f:
    json.dump(data, f, indent=4)
