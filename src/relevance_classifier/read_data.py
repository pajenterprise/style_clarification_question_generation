import re
import csv
from constants import *
import unicodedata
from collections import defaultdict
import math


def normalize_string(s, max_len):
    words = s.split()
    s = ' '.join(words[:max_len])
    return s


def get_context(line, max_post_len):
    context = normalize_string(line, max_post_len-1)
    return context


def read_data(context_fname, question_fname, ids_fname, labels_fname,
              max_post_len, max_ques_len):
    ids = []
    for line in open(ids_fname, 'r').readlines():
        curr_id = line.strip('\n')
        ids.append(curr_id)

    print("Reading lines...")
    data = []
    i = 0
    for line in open(context_fname, 'r').readlines():
        context = get_context(line, max_post_len)
        data.append([ids[i], context, None, None])
        i += 1

    i = 0
    for line in open(question_fname, 'r').readlines():
        question = normalize_string(line, max_ques_len-1)
        data[i][2] = question
        i += 1

    i = 0
    for line in open(labels_fname, 'r').readlines():
        label = int(line.strip('\n'))
        data[i][3] = label
        i += 1

    return data
