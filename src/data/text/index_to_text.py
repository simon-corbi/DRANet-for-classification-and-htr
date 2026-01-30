def convert_int_to_chars(indices, char_list):
    chars_sequence = ""

    for char_index in indices:
        try:
            c = char_list[char_index]
            chars_sequence += c
        except Exception as e:
            chars_sequence += "Error char index"

    return chars_sequence
