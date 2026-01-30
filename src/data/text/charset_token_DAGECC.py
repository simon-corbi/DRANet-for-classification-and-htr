
class CharsetToken(object):
    """ Contain all characters
    """

    def __init__(self, charset_file):
        self.charset_dictionary = {}
        self.charset_list = []
        self.char_number = 0

        with open(charset_file, mode='r', encoding="utf-8") as f:
            for line in f.readlines():
                if len(line) > 0:
                    c = line[:-1]
                    self.charset_dictionary[c] = self.char_number
                    self.charset_list.append(c)
                    self.char_number += 1

    def get_charset_dictionary(self):
        return self.charset_dictionary

    def get_charset_list(self):
        return self.charset_list

    def get_nb_char(self):
        return len(self.charset_list)

    def add_label(self, str_label):
        self.charset_dictionary[str_label] = self.char_number
        self.charset_list.append(str_label)
        self.char_number += 1
