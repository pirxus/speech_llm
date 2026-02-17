#!/bin/bash

cd /mnt/matylda2/data/MLC-SLM/MLC-SLM_Workshop-Training_Set_1/data/Spanish
# Initialize total duration
total_seconds=0

# Find all .wav files in subdirectories
# -quiet: don't show banner
# -show_entries: only grab the duration
# -of default=noprint_wrappers=1:nokey=1: output only the raw number
while read -r duration; do
    total_seconds=$(echo "$total_seconds + $duration" | bc)
done < <(find . -type f -name "*.wav" -exec ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {} \;)

# Convert seconds into a readable format (HH:MM:SS)
printf "Total Duration: %02d:%02d:%02d\n" $(echo "$total_seconds/3600" | bc) $(echo "($total_seconds%3600)/60" | bc) $(echo "$total_seconds%60" | bc)
