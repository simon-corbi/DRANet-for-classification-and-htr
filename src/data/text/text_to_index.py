def convert_text_to_index(dictionary, str_to_transform):
    """
    """

    labels = []

    for c in str_to_transform:
        if c not in dictionary:
            print(str_to_transform)
            print("Text unknow char in dictionnary: " + str(c))
            print("Ignore")
            continue
            # return -1
        else:
            labels.append(dictionary.get(c))

    return labels


def convert_index_to_chars(indices, char_list):
    """
    """

    chars_sequence = ""

    for char_index in indices:
        try:
            c = char_list[char_index]

            chars_sequence += c
        except Exception as e:
            chars_sequence += "Error char index"

    return chars_sequence
