import argparse
import csv
import sys
from collections import defaultdict


def main(args):
    ids = [line.strip('\n') for line in open(args.test_ids, 'r').readlines()]
    questions = [line.strip('\n') for line in open(args.test_question, 'r').readlines()]
    i = 0
    generic_ques = defaultdict(list)
    specific_ques = defaultdict(list)
    contexts = {}
    for line in open(args.test_context_with_labels, 'r').readlines():
        label, context = line.strip().split(" ", 1)
        contexts[ids[i]] = context
        if label == '<GENERIC>':
            generic_ques[ids[i]].append(questions[i])
        elif label == '<SPECIFIC>':
            specific_ques[ids[i]].append(questions[i])
        else:
            print('error!')
            return
        i += 1
    pred_ids = [line.strip('\n') for line in open(args.pred_test_question_ids, 'r').readlines()]
    pred_generic = [line.strip('\n') for line in open(args.pred_test_question_togeneric, 'r').readlines()]
    pred_specific = [line.strip('\n') for line in open(args.pred_test_question_tospecific, 'r').readlines()]
    pred_seq2seq = [line.strip('\n') for line in open(args.pred_test_question_seq2seq, 'r').readlines()]
    with open(args.analysis_output_file, 'w') as csvfile:
        fieldnames = ['asin', 'context', 'true_generic', 'pred_generic', 'true_specific', 'pred_specific', 'pred_seq2seq']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(pred_ids)):
            writer.writerow({'asin': pred_ids[i],
                             'context': contexts[pred_ids[i]],
                             'true_generic': ",".join(generic_ques[pred_ids[i]]),
                             'pred_generic': pred_generic[i],
                             'true_specific': ",".join(specific_ques[pred_ids[i]]),
                             'pred_specific': pred_specific[i],
                             'pred_seq2seq': pred_seq2seq[i]})


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--test_context_with_labels", type = str)
    argparser.add_argument("--test_question", type = str)
    argparser.add_argument("--test_ids", type=str)
    argparser.add_argument("--pred_test_question_ids", type=str)
    argparser.add_argument("--pred_test_question_seq2seq", type = str)
    argparser.add_argument("--pred_test_question_togeneric", type = str)
    argparser.add_argument("--pred_test_question_tospecific", type = str)
    argparser.add_argument("--analysis_output_file", type=str)
    args = argparser.parse_args()
    print(args)
    print("")
    main(args)
