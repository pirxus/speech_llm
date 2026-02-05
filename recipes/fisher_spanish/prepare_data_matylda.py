import librosa
import json
import os
from glob import glob



audio_dir = '/mnt/matylda2/data/SPANISH/LDC2010S01/data/speech'
transcripts_dir = '/mnt/matylda2/data/SPANISH/LDC2010T04/data/transcripts'

sph_files = glob(audio_dir + '/*.sph')
transcripts = glob(transcripts_dir + '/*.tdf')

# format: ['file;unicode', 'channel;int', 'start;float', 'end;float', 'speaker;unicode', 'speakerType;unicode', 'speakerDialect;unicode', 'transcript;unicode', 'section;int', 'turn;int', 'segment;int', 'sectionType;unicode', 'suType;unicode']
transcript_dict = {}

for transcript_path in transcripts:

    with open(transcript_path, 'r') as f:
        transcript = [ line.strip().split('\t') for line in f.readlines() ]
        transcript = transcript[3:]  # skip header

    for entry in transcript:
        file_id = os.path.basename(entry[0]).replace('.sph', '')
        if file_id not in transcript_dict:
            transcript_dict[file_id] = []
        transcript_dict[file_id].append({
            'channel': int(entry[1]),
            'start': float(entry[2]),
            'end': float(entry[3]),
            'length': float(entry[3]) - float(entry[2]),
            'speaker': entry[4],
            'transcript': entry[7],
            'section': int(entry[8]),
            'turn': int(entry[9]),
            'segment': int(entry[10]),
        })

# now, let's assemble the segments
conversations = {}
for file_id, data in transcript_dict.items():
    segments = []
    curr_segment = []

    curr_channel = data[0]['channel']
    curr_segment_id = data[0]['segment']
    for item in data:
        if item['transcript'].strip() == '<background>  </background>': # FIXME, empty segment lists
            continue
        if item['channel'] == curr_channel and curr_segment_id == item['segment']:
            curr_segment.append(item)
        else:
            segments.append(curr_segment)
            curr_segment = [item]
            curr_channel = item['channel']

            # check that the total segment duration is less than 30s
            assert segments[-1][-1]['end'] - segments[-1][0]['start'] < 30.0, print(json.dumps(segments[-1], indent=2))

    conversations[file_id] = segments

print(json.dumps(conversations, indent=2))


