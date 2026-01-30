import json
import glob
import os


def gather_labels_one_file(list_dir_label, path_save):
    dict_label = {}

    for one_d in list_dir_label:
        files_txt = glob.glob(one_d + '/*.txt')  # , recursive=True

        for one_file in files_txt:
            # Get id file
            split_name = os.path.split(one_file)
            split_name = split_name[1].split(sep=".")  # Filename and extension
            id_file = split_name[0]

            txt = ""

            # encoding="utf-8 Use for letter with diacritics
            with open(one_file, encoding="utf-8") as file:  # mode="r
                # with open(path_file, encoding="utf-8", mode="rb") as file:  # mode="r
                all_lines = file.readlines()

                for one_l in all_lines:
                    # if remove_line_break:
                    #     one_l = one_l.replace("\n", "")

                    txt += one_l

            if id_file not in dict_label:
                dict_label[id_file] = txt
            else:
                print("Incoherence: two labels with the same id " + id_file)

    print("Nb labels: " + str(len(dict_label)))

    json_object = json.dumps(dict_label, indent=4)

    with open(path_save, "w") as outfile:
        outfile.write(json_object)






