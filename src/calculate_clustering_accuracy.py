import argparse
import csv
import nltk
import sys
from collections import defaultdict


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
    lines = open(args.clustering_annotations).readlines()
    s_correct = 0
    s_wrong = 0
    g_correct = 0
    g_wrong = 0
    ques_clusters = defaultdict(list)
    good_clusters = []
    curr_cluster_id = None
    for i in range(5, len(lines)):
        line = lines[i]
        if line.strip('\n').strip() == '':
            continue
        elif ':' in line:
            cluster_id, ques_id, ques = line.split(':', 2)
            cluster_id = cluster_id.strip()
            ques_id = ques_id.strip()
            ques = ques.strip()
            curr_cluster_id = cluster_id
            ques_clusters[cluster_id].append((ques_id, ques))
        elif 'GOOD CLUSTER' in line:
            good_clusters.append(curr_cluster_id)

    for cluster_id in ques_clusters:
        is_generic = False
        is_specific = False
        if cluster_id in good_clusters:
            if len(ques_clusters[cluster_id]) > 10:
                #Clustering identifies questions in this cluster to be generic
                is_generic = True
            else:
                # Clustering identifies questions in this cluster to be specific
                is_specific = True
        else:
            #Clustering identifies questions in this cluster to be generic
            is_generic = True
        for ques_id, ques in ques_clusters[cluster_id]:
            if ques in generic_ques and ques_id == generic_ques_ids[generic_ques.index(ques)] and is_generic:
                g_correct += 1
            elif ques in generic_ques and ques_id == generic_ques_ids[generic_ques.index(ques)]and is_specific:
                g_wrong += 1
            elif ques in specific_ques and ques_id == specific_ques_ids[specific_ques.index(ques)]and is_specific:
                s_correct += 1
            elif ques in specific_ques and ques_id == specific_ques_ids[specific_ques.index(ques)]and is_generic:
                s_wrong += 1

    print('%d\t%d' % (s_correct, s_wrong))
    print('%d\t%d' % (g_wrong, g_correct))
    print('Total: %d' % (g_correct+g_wrong+s_correct+s_wrong))
    print('%.4f' % ((s_correct+g_correct)*1.0/(s_correct+g_correct+s_wrong+g_wrong)))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--human_annotations", type=str)
    argparser.add_argument("--clustering_annotations", type=str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)