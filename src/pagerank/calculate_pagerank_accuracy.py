import argparse
import csv
import nltk
import sys


def preprocess(text):
    text = text.replace('|', ' ')
    text = text.replace('/', ' ')
    text = text.replace('\\', ' ')
    text = text.lower()
    #text = re.sub(r'\W+', ' ', text)
    ret_text = ''
    for sent in nltk.sent_tokenize(text):
        ret_text += ' '.join(nltk.word_tokenize(sent)) + ' '
    return ret_text


def main(args):
    specific_ques_ids = []
    specific_ques = []
    generic_ques_ids = []
    generic_ques = []
    threshold = 0.00001
    with open(args.human_annotations) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ques_id = row['ident'].strip()
            ques = preprocess(row['question'].strip()).strip()
            #print(ques)
            if int(row['annotation_score']) in [3, 4] or row['annotation_category'] == 's':
                specific_ques_ids.append(ques_id)
                specific_ques.append(ques)
            elif row['annotation_score'] in [1, 2] or row['annotation_category'] == 'g':
                generic_ques_ids.append(ques_id)
                generic_ques.append(ques)
    specific_ques = specific_ques[:500]
    specific_ques_ids = specific_ques_ids[:500]
    generic_ques = generic_ques[:500]
    generic_ques_ids = generic_ques_ids[:500]
    #print(specific_ques_ids)
    lines = open(args.pagerank_annotations).readlines()
    s_correct = 0
    s_wrong = 0
    g_correct = 0
    g_wrong = 0
    for i in range(0, len(lines), 4):
        ques_id = lines[i].strip('\n').strip().split('_')[0]
        ques = lines[i+1].strip('\n').strip()
        ques_prob = float(lines[i+2].strip('\n').strip())
        if ques_id in specific_ques_ids and ques == specific_ques[specific_ques_ids.index(ques_id)]:
            if ques_prob < threshold:
                s_correct += 1
            else:
                s_wrong += 1
        if ques_id in generic_ques_ids and ques == generic_ques[generic_ques_ids.index(ques_id)]:
            if ques_prob >= threshold:
                g_correct += 1
            else:
                g_wrong += 1
    print('%d\t%d' % (s_correct, s_wrong))
    print('%d\t%d' % (g_wrong, g_correct))
    print('Total: %d' % (g_correct + g_wrong + s_correct + s_wrong))
    print('%.4f' % ((s_correct+g_correct)*1.0/(s_correct+g_correct+s_wrong+g_wrong)))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--human_annotations", type=str)
    argparser.add_argument("--pagerank_annotations", type=str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)