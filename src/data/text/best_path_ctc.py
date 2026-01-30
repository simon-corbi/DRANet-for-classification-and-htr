from src.data.text.text_to_index import convert_index_to_chars


def ctc_best_path_one(index_class, char_list, token_blank):
    # Remove the duplicated characters index
    sequence_without_duplicates = []
    previous_index = -1
    for index in index_class:
        if index != previous_index:
            sequence_without_duplicates.append(index)
            previous_index = index

    # Remove the blanks
    sequence = []
    for index in sequence_without_duplicates:
        if index != token_blank:
            sequence.append(index)

    # Convert to characters
    char_sequence = convert_index_to_chars(sequence, char_list)

    return char_sequence
