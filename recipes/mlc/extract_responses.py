import os
from glob import glob
from extraction_utils import *
import json
import uuid

BASE_PATH = "/mnt/matylda3/isvecjan/workspace/rustyspoons/tasks/spotify_dataset/exp_eloq"
WORK_DIR = "/mnt/matylda6/isedlacek/projects/eloquence/t2.5/speech_llm/recipes/mlc"

files = glob(os.path.join(BASE_PATH, "*.txt"))
files_base_names = [os.path.basename(f) for f in files]
data = {}

for split in ["dev", "train"]:
    # prepare the data structure for this split
    data[split] = {}
    data[split]['English/American'] = {}
    data[split]['Spanish'] = {}

    if split == "dev":
        files_to_process = [f for f, f_ in zip(files, files_base_names) if "dev" in f_]
    else:
        files_to_process = [f for f, f_ in zip(files, files_base_names) if "dev" not in f_]

    for file in files_to_process:
        print()
        print(f"Processing file: {file}")
        q_type = os.path.basename(file).split('.')[2]

        result = parse_processing_log_file(file)

        for key, value in result.items():
            language = key.rpartition('/')[0]
            conv_id = key.rpartition('/')[2]

            value = value.split("dialogs processed\n")[0].rpartition('\n')[0].strip()
            # let's try to parse the value into json lines
            try:
                json_lines = [json.loads(line) for line in value.split('\n') if line.strip()]

                # now check if each item in the json lines list adheres to the expected format based on the q_type
                for json_line in json_lines:
                    if q_type == "extractive":
                        # expected format: {"lang_id": string, "label": string, "task": string, "question": string, "answer": list[list[string]]}
                        assert "lang_id" in json_line and isinstance(json_line["lang_id"], str)
                        assert "label" in json_line and isinstance(json_line["label"], str)
                        assert "task" in json_line and isinstance(json_line["task"], str)
                        assert "question" in json_line and isinstance(json_line["question"], str)
                        assert "answer" in json_line and isinstance(json_line["answer"], list)
                        for ans in json_line["answer"]:
                            assert isinstance(ans, list)
                            for a in ans:
                                assert isinstance(a, str)

                    elif q_type == "abstractive":
                        # {"lang_id": string, "label": string, "task": string, "question": string, "answer": "string"}
                        assert "lang_id" in json_line and isinstance(json_line["lang_id"], str)
                        assert "label" in json_line and isinstance(json_line["label"], str)
                        assert "task" in json_line and isinstance(json_line["task"], str)
                        assert "question" in json_line and isinstance(json_line["question"], str)
                        assert "answer" in json_line and isinstance(json_line["answer"], str)

                    elif q_type == "impossible":
                        # {"lang_id": string, "label": string, "task": string, "question": string}
                        assert "lang_id" in json_line and isinstance(json_line["lang_id"], str)
                        assert "label" in json_line and isinstance(json_line["label"], str)
                        assert "task" in json_line and isinstance(json_line["task"], str)
                        assert "question" in json_line and isinstance(json_line["question"], str)


                # remove unnecessary items from the data structure
                for i, item in enumerate(json_lines):
                    if "lang_id" in item:
                        del item["lang_id"]
                    if "label" in item:
                        del item["label"]
                    if "task" in item:
                        del item["task"]
                    item['index'] = i
                    item['uuid'] = str(uuid.uuid4())

                # add conv_id to the data structure
                if conv_id not in data[split][language]:
                    data[split][language][conv_id] = {}

                data[split][language][conv_id][q_type] = json_lines

            except (json.JSONDecodeError, AssertionError):
                print(split, key.rpartition('/')[0], q_type, key.rpartition('_')[0].split('/')[-1])


print("Data processing complete. Summary:")
# let's print the keys of the data dictionary and the number of items in each split
for split in data:
    print(f"Split: {split}")
    for language in data[split]:
        print(f"  Language: {language}, Number of conversations: {len(data[split][language])}")

# let's save the data dictionary to a json file
with open(os.path.join(WORK_DIR, "processed_QAs.json"), "w") as f:
    json.dump(data, f, indent=4)

# let's save only the Spanish data
spanish_data = {split: data[split]['Spanish'] for split in data}
with open(os.path.join(WORK_DIR, "processed_QAs_spanish.json"), "w") as f:
    json.dump(spanish_data, f, indent=4)
