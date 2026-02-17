#!/usr/bin/env python3
"""
Script to simplify processed_QAs.json for recording purposes.
- Dev split: Keep all questions from all 5 conversations
- Train split: Randomly sample questions with ratio (abstractive + extractive)/impossible = 6/1
  Goal: At least one question per conversation
"""

import json
import random
from collections import defaultdict

def simplify_questions(input_file, output_file, seed=42):
    """
    Simplify the processed_QAs.json file for recording.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to the output JSON file
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Load the data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    simplified = {
        'dev': [],
        'train': []
    }
    
    # Process DEV split - keep all questions from all conversations (Spanish only)
    print("Processing DEV split...")
    dev_count = {'abstractive': 0, 'extractive': 0, 'impossible': 0}
    
    for accent, conversations in data['dev'].items():
        if accent != 'Spanish':
            continue
        for conv_id, questions in conversations.items():
            for q_type in ['abstractive', 'extractive', 'impossible']:
                if q_type in questions:
                    for q in questions[q_type]:
                        simplified['dev'].append({
                            'uuid': q['uuid'],
                            'question': q['question'],
                            'type': q_type,
                            'conversation_id': conv_id,
                            'accent': accent
                        })
                        dev_count[q_type] += 1
    
    print(f"DEV: {len(simplified['dev'])} questions total")
    print(f"  - Abstractive: {dev_count['abstractive']}")
    print(f"  - Extractive: {dev_count['extractive']}")
    print(f"  - Impossible: {dev_count['impossible']}")
    
    # Process TRAIN split - sample with ratio constraint
    print("\nProcessing TRAIN split...")
    
    # First, collect all questions by conversation
    conv_questions = defaultdict(lambda: {'abstractive': [], 'extractive': [], 'impossible': []})
    
    for accent, conversations in data['train'].items():
        if accent != 'Spanish':
            continue
        for conv_id, questions in conversations.items():
            for q_type in ['abstractive', 'extractive', 'impossible']:
                if q_type in questions:
                    for q in questions[q_type]:
                        conv_questions[conv_id][q_type].append({
                            'uuid': q['uuid'],
                            'question': q['question'],
                            'type': q_type,
                            'conversation_id': conv_id,
                            'accent': accent
                        })
    
    print(f"Total conversations in TRAIN: {len(conv_questions)}")
    
    # Sample questions with the constraint: (abstractive + extractive)/impossible = 6/1
    # Strategy: For each conversation, randomly select questions
    # Then adjust globally to meet the ratio
    
    selected_questions = []
    
    for conv_id, questions in conv_questions.items():
        # Try to select at least one question per conversation
        # Prioritize abstractive and extractive (6:1 ratio)
        
        available = []
        # Add abstractive and extractive with higher weight
        available.extend([(q, 6) for q in questions['abstractive']])
        available.extend([(q, 6) for q in questions['extractive']])
        # Add impossible with lower weight
        available.extend([(q, 1) for q in questions['impossible']])
        
        if available:
            # Weighted random selection - pick one question per conversation
            weights = [w for _, w in available]
            selected = random.choices([q for q, _ in available], weights=weights, k=1)[0]
            selected_questions.append(selected)
    
    # Count the selected questions by type
    train_count = {'abstractive': 0, 'extractive': 0, 'impossible': 0}
    for q in selected_questions:
        train_count[q['type']] += 1
    
    # Check if we need to adjust to meet the 6:1 ratio
    total_ae = train_count['abstractive'] + train_count['extractive']
    total_imp = train_count['impossible']
    
    print(f"\nInitial selection: {len(selected_questions)} questions")
    print(f"  - Abstractive: {train_count['abstractive']}")
    print(f"  - Extractive: {train_count['extractive']}")
    print(f"  - Impossible: {train_count['impossible']}")
    print(f"  - Ratio (A+E)/I: {total_ae}/{total_imp} = {total_ae/total_imp if total_imp > 0 else 'inf'}")
    
    # Adjust if ratio is not 6:1
    # Target: for every 6 (abstractive + extractive), have 1 impossible
    target_ratio = 6.0
    
    if total_imp > 0:
        current_ratio = total_ae / total_imp
        
        # If we have too many impossible questions, remove some
        if current_ratio < target_ratio:
            target_impossible = int(total_ae / target_ratio)
            impossible_questions = [q for q in selected_questions if q['type'] == 'impossible']
            questions_to_remove = random.sample(impossible_questions, total_imp - target_impossible)
            selected_questions = [q for q in selected_questions if q not in questions_to_remove]
            print(f"\nRemoved {len(questions_to_remove)} impossible questions to meet ratio")
        
        # If we have too few impossible questions, add some
        elif current_ratio > target_ratio:
            target_impossible = int(total_ae / target_ratio)
            # Collect all impossible questions not yet selected
            all_impossible = []
            for conv_id, questions in conv_questions.items():
                all_impossible.extend(questions['impossible'])
            
            # Remove already selected
            selected_uuids = {q['uuid'] for q in selected_questions}
            available_impossible = [q for q in all_impossible if q['uuid'] not in selected_uuids]
            
            # Add more impossible questions
            to_add = min(target_impossible - total_imp, len(available_impossible))
            if to_add > 0:
                additional = random.sample(available_impossible, to_add)
                selected_questions.extend(additional)
                print(f"\nAdded {to_add} impossible questions to meet ratio")
    
    # Final count
    train_count = {'abstractive': 0, 'extractive': 0, 'impossible': 0}
    for q in selected_questions:
        train_count[q['type']] += 1
    
    simplified['train'] = selected_questions
    
    total_ae = train_count['abstractive'] + train_count['extractive']
    total_imp = train_count['impossible']
    
    print(f"\nFinal TRAIN selection: {len(simplified['train'])} questions")
    print(f"  - Abstractive: {train_count['abstractive']}")
    print(f"  - Extractive: {train_count['extractive']}")
    print(f"  - Impossible: {train_count['impossible']}")
    print(f"  - Final ratio (A+E)/I: {total_ae}/{total_imp} = {total_ae/total_imp if total_imp > 0 else 'inf'}")
    
    # Count unique conversations covered
    unique_convs = len(set(q['conversation_id'] for q in selected_questions))
    print(f"  - Unique conversations covered: {unique_convs}/{len(conv_questions)}")
    
    # Save the simplified data
    with open(output_file, 'w') as f:
        json.dump(simplified, f, indent=2)
    
    print(f"\nSimplified data saved to: {output_file}")

if __name__ == '__main__':
    simplify_questions('processed_QAs.json', 'simplified_QAs.json')