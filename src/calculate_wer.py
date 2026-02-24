import json
from english_normalizer import EnglishNormalizer
from jiwer import compute_measures
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, nargs="+", help="List of prediction json files.")
parser.add_argument("--includes", type=str, default="", help="Filter only experiments that include this string in the name.")

args = parser.parse_args()

normalizer = EnglishNormalizer()
def normalize(text):
    return normalizer(text).strip()

def pprint(string):
    print(json.dumps(string, indent=4))

def calculate_wer(gt, hyp):
    metrics = compute_measures(gt, hyp)
    del metrics["ops"]
    del metrics["truth"]
    del metrics["hypothesis"]
    del metrics["wil"]
    del metrics["mer"]
    del metrics["wip"]
    return metrics


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
        
        labels = [ normalize(item) for item in split_data['labels'] ]
        predictions = [ normalize(item) for item in split_data['predictions'] ]

        metrics = calculate_wer(labels, predictions)
        print(f"Metrics for split {split}:")
        pprint(metrics)
        print("=" * 100)
