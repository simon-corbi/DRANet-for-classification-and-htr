import glob


def create_charset(dir_data, path_save):
    """
    Read all label ground truth and create a file with all characters present
    """
    files = glob.glob(dir_data + '/**/*.txt', recursive=True)

    full_text = ""

    for one_file_label in files:
        label = ""
        with open(one_file_label, "r", encoding="utf-8") as file:
            label = file.readline()

        full_text += label

    charset = set(full_text)
    charset = sorted(charset)

    with open(path_save, 'w', encoding="utf-8") as file:
        for one_char in charset:
            file.write(one_char)
            file.write("\n")


def merge_charset(list_path_charset, path_save_merge):
    charset_dictionary = {}
    char_number = 0

    for path_c in list_path_charset:
        with open(path_c, mode='r', encoding="utf-8") as f:
            for line in f.readlines():
                if len(line) > 0:
                    c = line[:-1]

                    if c not in charset_dictionary:
                        charset_dictionary[c] = char_number
                        char_number += 1

    myKeys = list(charset_dictionary.keys())
    myKeys.sort()
    sorted_dict = {i: charset_dictionary[i] for i in myKeys}
    print("charset_dict:")
    print(sorted_dict)

    with open(path_save_merge, 'w', encoding="utf-8") as file:
        for one_class in sorted_dict:
            file.write(one_class)
            file.write("\n")
