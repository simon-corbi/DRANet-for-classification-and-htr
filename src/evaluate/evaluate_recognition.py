from enum import Enum
import re

import editdistance

# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize


class ProcessWER(Enum):
    NO = 1
    DAS_V2 = 2
    DAN = 3

    def __str__(self):
        return self.name


def nb_chars_from_list(list_gt):
    return sum([len(t) for t in list_gt])


def nb_words_from_list(list_gt, format_string=ProcessWER.NO):
    len_ = 0
    for gt in list_gt:
        # if format_string == ProcessWER.DAS_V2:
        #     gt = word_tokenize(gt)
        # el
        if format_string == ProcessWER.DAN:
            gt = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', gt)  # punctuation processed as word
            gt = gt.split(" ")
        else:
            gt = gt.split(" ")
        len_ += len(gt)
    return len_


def edit_wer_from_list(truth, pred, format_string=ProcessWER.NO):
    edit = 0
    for pred, gt in zip(pred, truth):
        # if format_string == ProcessWER.DAS_V2:
        #     gt = word_tokenize(gt)
        #     pred = word_tokenize(pred)
        # el
        if format_string == ProcessWER.DAN:
            gt = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', gt)  # punctuation processed as word
            pred = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', pred)  # punctuation processed as word

            gt = gt.split(" ")
            pred = pred.split(" ")
        else:
            gt = gt.split(" ")
            pred = pred.split(" ")
        edit += editdistance.eval(gt, pred)
    return edit


def nb_words_one(gt, format_string=ProcessWER.NO):
    len_ = 0

    # if format_string == ProcessWER.DAS_V2:
    #     gt = word_tokenize(gt)
    # el
    if format_string == ProcessWER.DAN:
        gt = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', gt)  # punctuation processed as word
        gt = gt.split(" ")
    else:
        gt = gt.split(" ")
    len_ += len(gt)

    return len_


def edit_wer_one(gt, pred, format_string=ProcessWER.NO):
    edit = 0

    # if format_string == ProcessWER.DAS_V2:
    #     gt = word_tokenize(gt)
    #     pred = word_tokenize(pred)
    # el
    if format_string == ProcessWER.DAN:
        gt = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', gt)  # punctuation processed as word
        pred = re.sub('([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', pred)  # punctuation processed as word

        gt = gt.split(" ")
        pred = pred.split(" ")
    else:
        gt = gt.split(" ")
        pred = pred.split(" ")
    edit += editdistance.eval(gt, pred)

    return edit
