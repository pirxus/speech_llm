def parse_processing_log(content):
    """
    Parse log content and split it by 'Processing ...' lines.

    This function takes log content where each dialog section starts with a line
    like "Processing English/American/0517_007..." and extracts the dialog ID
    along with all content until the next "Processing" line.

    Args:
        content (str): String content of the log file

    Returns:
        dict: Dictionary with structure {id: chunk_content} where:
              - id: Dialog identifier extracted from the Processing line
                    (e.g., 'English/American/0517_007')
              - chunk_content: All lines between this Processing line and the next,
                              joined with newlines

    Example:
        >>> content = '''Processing English/American/0517_007...
        ... {"lang_id": "English/American", "label": "0517_007"}
        ... Progress: 1/14 dialogs processed
        ... Processing English/American/0525_002...
        ... {"lang_id": "English/American", "label": "0525_002"}'''
        >>> result = parse_processing_log(content)
        >>> len(result)
        2
        >>> 'English/American/0517_007' in result
        True
    """
    #lines = content.split('\n')[5:]
    lines = content.partition("dialogs for task")[2].split('\n')[1:]

    result = {}
    current_id = None
    current_chunk = []

    for line in lines:
        if line.startswith('Processing '):
            # Save previous chunk if exists
            if current_id is not None:
                result[current_id] = '\n'.join(current_chunk)

            # Extract the ID (everything after "Processing " and before "...")
            # Remove "Processing " prefix and trailing periods
            id_part = line.replace('Processing ', '').rstrip('.')
            current_id = id_part
            current_chunk = []
        else:
            # Add line to current chunk
            current_chunk.append(line)

    # Don't forget the last chunk
    if current_id is not None:
        result[current_id] = '\n'.join(current_chunk)

    return result


def parse_processing_log_file(file_path):
    """
    Parse a log file and split it by 'Processing ...' lines.

    Args:
        file_path (str): Path to the log file

    Returns:
        dict: Dictionary with structure {id: chunk_content}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_processing_log(content)
