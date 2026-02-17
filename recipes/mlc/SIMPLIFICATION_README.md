# Question Simplification for Recording

## Overview

This document describes the simplification process applied to `processed_QAs.json` to create `simplified_QAs.json` for recording purposes.

**Note:** Only Spanish questions are included in the simplified file.

## Process

The simplification was done using `simplify_questions.py` script with the following specifications:

### Dev Split
- **All Spanish questions included** from all Spanish conversations
- **Total:** 75 questions
  - Abstractive: 25
  - Extractive: 25
  - Impossible: 25
- **Strategy:** Complete coverage for validation purposes (Spanish only)
- **Recording approach:** Questions can be randomly distributed among speakers (not requiring multiple speakers per question to save time)

### Train Split
- **Goal:** At least one question per conversation (all 270 conversations covered)
- **Total:** 282 questions
  - Abstractive: 128
  - Extractive: 114
  - Impossible: 40
- **Ratio constraint:** (Abstractive + Extractive) / Impossible ≈ 6:1
  - Achieved ratio: 242/40 = 6.05 ✓
- **Strategy:** 
  - Randomly sampled questions prioritizing abstractive and extractive types
  - Ensured coverage of all 270 conversations
  - Adjusted selection to meet the 6:1 ratio constraint
- **Recording approach:** One speaker per question, randomly distributed

## Output Format

The `simplified_QAs.json` file has a flat structure for easy distribution:

```json
{
  "dev": [
    {
      "uuid": "question-uuid",
      "question": "Question text",
      "type": "abstractive|extractive|impossible",
      "conversation_id": "conversation-id",
      "accent": "accent-category"
    },
    ...
  ],
  "train": [
    ...
  ]
}
```

## Key Features

1. **No answers included** - Only question metadata needed for recording
2. **UUID tracking** - Each question has a unique identifier for linking back to original data
3. **Conversation tracking** - `conversation_id` allows tracking which conversation each question belongs to
4. **Accent information** - Preserved for potential speaker assignment considerations
5. **Type labels** - Clear categorization as abstractive, extractive, or impossible

## Priority

1. **First priority:** Dev split (75 Spanish questions)
2. **Second priority:** Train split (282 Spanish questions covering all 270 Spanish conversations)

## Usage

To regenerate the simplified file:

```bash
python3 simplify_questions.py
```

The script uses a fixed random seed (42) for reproducibility.