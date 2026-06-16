import json
import re
import sys
import argparse
import torch
from english_normalizer import EnglishNormalizer
from jiwer import compute_measures
from rouge_score import rouge_scorer

QUESTION_CATEGORIES = ['extractive', 'abstractive', 'impossible']

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, nargs="+", help="List of prediction json files.")
parser.add_argument("--includes", type=str, default="", help="Filter only experiments that include this string in the name.")
parser.add_argument("--bert_score", action="store_true", help="Compute BERTScore on answers (slow, requires bert-score package).")
parser.add_argument("--data_dir", type=str, default="/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/recipes/mlc/data",
                    help="Path to the dataset directory (used for category-wise evaluation when uuids are present).")
parser.add_argument("--verbose", action="store_true", help="Print parse failures to stderr.")
parser.add_argument("--repair", action="store_true", help="Attempt to repair malformed JSON in predictions/labels (handles unescaped quotes, missing value quotes, trailing commas).")
parser.add_argument("--rephrased_extractive", type=str,
                    default="/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/exp/data/rephrased_answers.json",
                    help="Path to rephrased extractive answers JSON. When provided, loads original extractive "
                         "answers for Answer Coverage Score computation.")

args = parser.parse_args()

normalizer = EnglishNormalizer()
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.bert_score:
    from bert_score import score as bert_score_fn

def normalize(text):
    return normalizer(text).strip()

def normalize_for_coverage(text):
    """Simple lowercase + punctuation removal for Answer Coverage Score."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    return text.strip()

def pprint(string):
    print(json.dumps(string, indent=4))

def repair_json(s):
    """
    Attempt to fix common JSON encoding errors produced by the model:
      1. Unescaped double quotes inside string values.
      2. Missing opening quote on a string value  ("key": value  ->  "key": "value").
      3. Trailing comma before closing brace  (...,}  ->  ...}).
    Uses a character-level state machine for (1).
    """
    # Pass 1: escape unescaped inner quotes using a lookahead heuristic.
    # When inside a string and we encounter '"', peek ahead past whitespace:
    # if the next character is one of  :  ,  }  ]  the quote legitimately
    # closes the string; otherwise it is an unescaped inner quote -- escape it.
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_string:
            # already-escaped character -- pass through both chars unchanged
            result.append(c)
            if i + 1 < len(s):
                result.append(s[i + 1])
            i += 2
            continue
        if c == '"':
            if not in_string:
                in_string = True
                result.append(c)
            else:
                j = i + 1
                while j < len(s) and s[j] in ' \t\r\n':
                    j += 1
                if j >= len(s) or s[j] in ':,}]':
                    in_string = False
                    result.append(c)
                else:
                    result.append('\\"')
        else:
            result.append(c)
        i += 1
    s = ''.join(result)

    # Pass 2: add missing opening quote on bare string values
    # e.g.  "answer": some text  ->  "answer": "some text
    s = re.sub(r'("(?:question|answer)"\s*:\s*)([^"\s{[])', r'\1"\2', s)

    # Pass 3: remove trailing comma before closing brace
    s = re.sub(r',\s*}', '}', s)

    return s

def parse_json_from_string(text):
    """Extract and parse the first JSON object found in a string.
    Falls back to a repair pass for common model output encoding errors."""
    # Use a greedy match to capture the full outermost {...} block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    candidate = match.group()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    if args.repair:
        try:
            return json.loads(repair_json(candidate))
        except json.JSONDecodeError:
            pass
    return None

def calculate_wer(gt, hyp):
    metrics = compute_measures(gt, hyp)
    del metrics["ops"]
    del metrics["truth"]
    del metrics["hypothesis"]
    del metrics["wil"]
    del metrics["mer"]
    del metrics["wip"]
    return metrics

def calculate_rouge_l(gt_list, hyp_list):
    scores = [scorer.score(g, h)['rougeL'].fmeasure for g, h in zip(gt_list, hyp_list)]
    return sum(scores) / len(scores) if scores else 0.0

def calculate_exact_match(gt_list, hyp_list):
    matches = sum(g.strip().lower() == h.strip().lower() for g, h in zip(gt_list, hyp_list))
    return matches / len(gt_list) if gt_list else 0.0

def calculate_bert_score(gt_list, hyp_list):
    _, _, F1 = bert_score_fn(hyp_list, gt_list, lang='en', device=device, verbose=False)
    return F1.mean().item()

def calculate_answer_coverage(answer_extractive_list, hyp_list):
    """
    Answer Coverage Score: fraction of required answer items (outer list) recovered
    in the prediction, where an item is recovered if any of its acceptable variants
    (inner list) appears as a substring in the normalized prediction.

    answer_extractive_list: list of list of list of str  (one per sample)
      outer list  = required items (all must be covered for full score)
      inner list  = acceptable variants for that item (any one suffices)
    hyp_list: list of str  (one normalized prediction per sample)
    """
    scores = []
    for answer_items, hyp in zip(answer_extractive_list, hyp_list):
        if not answer_items:
            scores.append(1.0)
            continue
        norm_hyp = normalize_for_coverage(hyp)
        n_correct = sum(
            any(normalize_for_coverage(variant) in norm_hyp for variant in variants)
            for variants in answer_items
        )
        scores.append(n_correct / len(answer_items))
    return sum(scores) / len(scores) if scores else 0.0

def calculate_impossible_metrics(samples):
    """
    Precision, recall and F1 for impossible question detection.
    A question is predicted impossible when the prediction JSON has no 'answer' key.
    Requires samples to have 'category' and 'pred_has_answer' fields.
    Returns None if category information is unavailable.
    """
    if any(s['category'] is None for s in samples):
        return None
    tp = sum(1 for s in samples if s['category'] == 'impossible' and not s['pred_has_answer'])
    fp = sum(1 for s in samples if s['category'] != 'impossible' and not s['pred_has_answer'])
    fn = sum(1 for s in samples if s['category'] == 'impossible' and s['pred_has_answer'])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision, 'recall': recall, 'f1': f1}

def print_metrics(questions_gt, questions_hyp, answers_gt, answers_hyp,
                  answer_extractive_list=None, impossible_metrics=None, indent="  "):
    if not questions_gt:
        print(f"{indent}(no samples)")
        return
    wer_metrics = calculate_wer(questions_gt, questions_hyp)
    print(f"{indent}Question WER metrics:")
    print(indent + json.dumps(wer_metrics, indent=4).replace('\n', '\n' + indent))
    rouge_l = calculate_rouge_l(answers_gt, answers_hyp)
    exact_match = calculate_exact_match(answers_gt, answers_hyp)
    print(f"{indent}Answer ROUGE-L:     {rouge_l:.4f}")
    print(f"{indent}Answer Exact Match: {exact_match:.4f}")
    if answer_extractive_list is not None:
        coverage = calculate_answer_coverage(answer_extractive_list, answers_hyp)
        print(f"{indent}Answer Coverage:    {coverage:.4f}")
    if args.bert_score:
        bs = calculate_bert_score(answers_gt, answers_hyp)
        print(f"{indent}Answer BERTScore F1: {bs:.4f}  (device: {device})")
    if impossible_metrics is not None:
        m = impossible_metrics
        print(f"{indent}Impossible detection -- "
              f"P: {m['precision']:.4f}  R: {m['recall']:.4f}  F1: {m['f1']:.4f}"
              f"  (TP={m['tp']} FP={m['fp']} FN={m['fn']})")

def load_uuid_maps(data_dir, rephrased_extractive):
    """Load SpanishQADataset (test split) and return uuid -> type and uuid -> answer_extractive dicts."""
    from qa.qa_dataset import SpanishQADataset
    print(f"  Loading SpanishQADataset (test split) from {data_dir} ...")
    dataset = SpanishQADataset(
        split='test',
        data_dir=data_dir,
        tokenizer=None,
        rephrased_extractive=rephrased_extractive,
    )
    uuid_type = {item['uuid']: item['type'] for item in dataset.data}
    uuid_extractive = {
        item['uuid']: item['answer_extractive']
        for item in dataset.data
        if 'answer_extractive' in item
    }
    return uuid_type, uuid_extractive, dataset


uuid_type_map = None
uuid_extractive_map = None
qa_dataset = None

for input_file in args.input:

    exp_name = input_file.split('/')[-2]

    if args.includes not in exp_name:
        continue
    try:
        print(f"Processing experiment {exp_name}")
    except:
        pass
    with open(input_file, 'r') as f:
        data = json.load(f)

    for split, split_data in data.items():

        raw_labels = split_data['labels']
        raw_predictions = split_data['predictions']
        raw_uuids = split_data.get('uuids', None)

        # Load dataset for category/extractive mapping if uuids are present (once per run)
        if raw_uuids is not None and uuid_type_map is None:
            uuid_type_map, uuid_extractive_map, qa_dataset = load_uuid_maps(
                args.data_dir, args.rephrased_extractive
            )

        # Parse predictions and labels, attaching uuid, category, and extractive answer where available
        samples = []
        n_parse_failures = 0

        for i, (label, pred) in enumerate(zip(raw_labels, raw_predictions)):
            label_json = parse_json_from_string(label)
            pred_json = parse_json_from_string(pred)

            if label_json is None or pred_json is None:
                n_parse_failures += 1
                if args.verbose:
                    print(f"  [{'label' if label_json is None else 'pred'} parse failure #{n_parse_failures}] index={i}", file=sys.stderr)
                    print(f"    label: {label!r}", file=sys.stderr)
                    print(f"    pred:  {pred!r}", file=sys.stderr)
                continue

            uuid = raw_uuids[i] if raw_uuids is not None else None
            category = uuid_type_map.get(uuid) if (uuid_type_map is not None and uuid is not None) else None
            answer_extractive = uuid_extractive_map.get(uuid) if (uuid_extractive_map is not None and uuid is not None) else None

            samples.append({
                'question_gt':       normalize(label_json.get('question', '')),
                'question_hyp':      normalize(pred_json.get('question', '')),
                'answer_gt':         normalize(label_json.get('answer', '')),
                'answer_hyp':        normalize(pred_json.get('answer', '')),
                'pred_has_answer':   'answer' in pred_json,
                'category':          category,
                'answer_extractive': answer_extractive,
            })

        print(f"Metrics for split {split}:")
        print(f"  Parsed: {len(samples)}/{len(raw_labels)}  (parse failures: {n_parse_failures})")

        def extract_lists(subset):
            extractive_list = [s['answer_extractive'] for s in subset if s['answer_extractive'] is not None]
            has_extractive = len(extractive_list) == len(subset) and len(subset) > 0
            return (
                [s['question_gt']  for s in subset],
                [s['question_hyp'] for s in subset],
                [s['answer_gt']    for s in subset],
                [s['answer_hyp']   for s in subset],
                extractive_list if has_extractive else None,
            )

        # Overall metrics
        print(f"\n  --- Overall ({len(samples)} samples) ---")
        overall_impossible = calculate_impossible_metrics(samples) if raw_uuids is not None else None
        print_metrics(*extract_lists(samples), impossible_metrics=overall_impossible)

        # Per-category metrics
        if raw_uuids is not None:
            for cat in QUESTION_CATEGORIES:
                cat_samples = [s for s in samples if s['category'] == cat]
                print(f"\n  --- Category: {cat} ({len(cat_samples)} samples) ---")
                # For the impossible category, compute detection metrics scoped to
                # just impossible vs. the rest of the full set (not just impossible samples)
                if cat == 'impossible':
                    imp_metrics = calculate_impossible_metrics(samples)
                else:
                    imp_metrics = None
                print_metrics(*extract_lists(cat_samples), impossible_metrics=imp_metrics)

        print("=" * 100)
