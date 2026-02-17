import time
import string
import os
import random
import json
from utils import load_api_keys, ClientCycler, generate_response
from google.genai import types
import sys

import argparse

random.seed(42)

parser = argparse.ArgumentParser(description="Evaluate Gemini model on audio files.")
parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file containing audio paths and questions.")
parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file to save the results.")
parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Name of the Gemini model to use.")
parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the audio files.")
parser.add_argument("--start_i", type=int, default=0, help="Index to start processing from in the dataset.")
parser.add_argument("--shuffle", action='store_true', help="Shuffle the dataset before processing.")

args = parser.parse_args()

api_keys = load_api_keys(path='/mnt/matylda6/isedlacek/projects/jsalt/mmau_pro/api_keys')
#api_keys = load_api_keys(path='/mnt/matylda6/isedlacek/projects/context_annotation/api_keys')
""" api keys file has a format like
%key1% # comment
%key2% # comment
"""
cycler = ClientCycler(api_keys)
client = cycler.get_next_client()

config = types.GenerateContentConfig(
    system_instruction="You are a helpful assistant that answers questions based on audio content. When given a list of options, only reply by choosing the correct answer from the options provided, do not add any extra information. If the question is open-ended, provide a concise answer based on the audio content.",
)

with open(args.input_json, "r") as f:
    input_json = json.load(f)

# filter out multiaudio
print(len(input_json))
input_json = list(filter(lambda x: isinstance(x['audio_path'], str), input_json))
print(len(input_json))

def get_input_prompt(sample):
    question = sample['question']
    choices = sample['choices']
    # take as many uppercase letters as there are choices
    choice_indices = list(string.ascii_uppercase[:len(choices)])
    # build the choice text
    choice_text = "\n".join(
        f"{idx}) {choice}"
        for idx, choice in zip(choice_indices, choices)
    )
    # assemble full prompt
    text_content = (
        f"{question}\n"
        f"Choose the correct answer from the following choices:\n"
        f"{choice_text}"
    )
    return text_content


answers = []
i = args.start_i
while i < len(input_json):
    try:
        item = input_json[i]
        audio_path = os.path.join(args.data_dir, item['audio_path'])

        choices = item['choices']

        if not isinstance(choices, list) or len(choices) in [0, 1]:
            # open ended
            prompt = item['question']

        else:
            if args.shuffle:
                random.shuffle(item['choices'])

            prompt = get_input_prompt(item)

        retry_counter = 0
        while True:
            try:
                print(i)
                response, myfile = generate_response(audio_path, prompt, client, config=config, model=args.model_name)
                break
            except Exception as e:
                print(e)
                retry_counter += 1
                if retry_counter > 10:
                    print("too many retries, getting a new client")
                    client = cycler.get_next_client()
                    if client is None:
                        print("No more clients available, exiting.")
                        break
                    else:
                        print("switched to a new client")
                        retry_counter = 0

        if retry_counter > 10:
            break

        item['model_output'] = response.text
        answers.append(item)

    except Exception as e:
        print(f"Error processing item at index {i}: {e}")
        print(item)

    i += 1

    client.files.delete(name=myfile.name)

    print(f"Processing item {i}, {item['id']}")
    print(prompt)
    print("Answer:", item['answer'])
    print('Model output:', response.text)
    print("==="*10)
    sys.stdout.flush()

if i < len(input_json):
    print(f"Stopped early at index {i} due to API limits or errors.")

with open(args.output_file, "w") as f:
    json.dump(answers, f, indent=4)
