from src.data.global_values.text_global_values import BLANK_STR_TOKEN


class CharsetToken(object):
    """ Contain all labels of an alphabet
    special character : blank (for ctc), Start Of Sequence (sos) and End Of Sequence (eos) for Seq2Seq
    """

    def __init__(self, list_charset_file,  use_blank=False):
        self.charset_dictionary = {}
        self.charset_list = []
        self.char_number = 0

        if use_blank:
            self.charset_dictionary[BLANK_STR_TOKEN] = 0
            self.charset_list.append(BLANK_STR_TOKEN)
            self.char_number += 1
        # Merge several charsets if there are more than one
        for one_charset_file in list_charset_file:
            with open(one_charset_file, mode='r', encoding="utf-8") as f:
                for line in f.readlines():
                    if len(line) > 0:
                        c = line[:-1]

                        if c not in self.charset_dictionary:
                            self.charset_dictionary[c] = self.char_number
                            self.charset_list.append(c)
                            self.char_number += 1

    def add_char(self, char):
        if char not in self.charset_dictionary:
            self.charset_dictionary[char] = self.char_number
            self.charset_list.append(char)
            self.char_number += 1

    def get_charset_dictionary(self):
        return self.charset_dictionary

    def get_charset_list(self):
        return self.charset_list

    def get_nb_char(self):
        return len(self.charset_list)
