import json
import os
from glob import glob

MLC_DEV_PATH = "/mnt/matylda2/data/MLC-SLM/MLC-SLM_Workshop-Development_Set/data"
SPLITS = ['English/American', 'Spanish']


def get_line_data(line):
    start, end, speaker, transcript = line.split('\t')

    return {
        'start': float(start),
        'end': float(end),
        'speaker': speaker,
        'transcript': transcript,
    }

final_data = {}
for split in SPLITS:

    data_path = os.path.join(MLC_DEV_PATH, split)
    txt_files = glob(data_path + "/*.txt")
    final_data[split] = {}

    for txt in txt_files:


        with open(txt, 'r') as f:
            lines = [ line.strip() for line in f.readlines() ]

        convo = []
        current_turn = []
        current_speaker = lines[0].split('\t')[2]
        turn = 0
        for line in lines:
            data = get_line_data(line)

            if current_speaker != data['speaker']:
                convo.append({
                    'turn': turn,
                    'speaker': current_speaker,
                    'transcript': current_turn,
                })

                current_speaker = data['speaker']
                current_turn = [data['transcript']]
                turn += 1

            else:
                current_turn.append(data['transcript'])
                
        final_data[split][os.path.basename(txt)[:-4]] = convo


with open('conversations_dev.json', 'w') as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)


# let's print a single conversation

dialogue = final_data['English/American']['0517_007']

speaker = 1
for item in dialogue:
    print(f"Turn {item['turn']}, Speaker {2 if speaker == 0 else speaker}: {' '.join(item['transcript'])}")
    speaker = int(not speaker)
