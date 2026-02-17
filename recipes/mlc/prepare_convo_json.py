import json
import os
from glob import glob

MLC_DEV_PATH = "/mnt/matylda2/data/MLC-SLM/MLC-SLM_Workshop-Development_Set/data"
MLC_TRAIN_PATH = "/mnt/matylda2/data/MLC-SLM/MLC-SLM_Workshop-Training_Set_1/data"
SPLITS = ['English/American', 'Spanish']
SETS = [MLC_DEV_PATH, MLC_TRAIN_PATH]

def get_line_data(line):
    start, end, speaker, transcript = line.split('\t')

    return {
        'start': float(start),
        'end': float(end),
        'speaker': speaker,
        'transcript': transcript,
    }


def stringify_conversation(conversation, split):
    SPLITS = ['English/American', 'Spanish']
    assert split in SPLITS, "Invalid split specified"
    if split == "English/American":
        turn_string = "Turn"
        speaker_string = "Speaker"
    else:
        turn_string = "Turno"
        speaker_string = "Hablante"

    speaker = 1
    dialogue = []
    for item in conversation:
        dialogue.append(f"{turn_string} {int(item['turn']) + 1}, {speaker_string} {2 if speaker == 0 else speaker}: {' '.join(item['transcript'])}")
        speaker = int(not speaker)

    return dialogue


for split_path in SETS:
    final_data = {}
    for split in SPLITS:

        data_path = os.path.join(split_path, split)
        if "Development" in split_path:
            txt_files = glob(data_path + "/*.txt")
        else:
            txt_files = glob(data_path + "/*/*.txt")
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


    if "Development" in split_path:
        with open('conversations_dev.json', 'w') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
    else:
        with open('conversations_train.json', 'w') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

    try:
        # let's print a single conversation
        split = 'English/American'
        dialogue = final_data[split][list(final_data[split].keys())[0]]
        convo = stringify_conversation(dialogue, split)
        print('\n'.join(convo))
    except:
        pass

    for split in list(final_data.keys()):
        print(f"{split}: {len(final_data[split].keys())}")
