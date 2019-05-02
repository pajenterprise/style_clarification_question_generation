from helper import *
import numpy as np
import torch
from constants import *


def evaluate(context_model, question_model, relevance_model, dev_data, criterion, args):
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0    

    context_model.eval()
    question_model.eval()
    relevance_model.eval()
    
    with torch.no_grad():
        ids, contexts, context_lens, questions, question_lens, labels = dev_data
        context_masks = get_masks(context_lens, args.max_post_len)
        question_masks = get_masks(question_lens, args.max_ques_len)
        contexts = np.array(contexts)
        questions = np.array(questions)
        labels = np.array(labels)
        for c, cm, q, qm, l in iterate_minibatches(contexts, context_masks,
                                                          questions, question_masks,
                                                          labels, args.batch_size):
            c = torch.LongTensor(c)
            cm = torch.FloatTensor(cm)
            q = torch.LongTensor(q)
            qm = torch.FloatTensor(qm)
            if USE_CUDA:
                c = c.cuda()
                cm = cm.cuda()
                q = q.cuda()
                qm = qm.cuda()

            c_hid, c_out = context_model(torch.transpose(c, 0, 1))
            cm = torch.transpose(cm, 0, 1).unsqueeze(2)
            cm = cm.expand(cm.shape[0], cm.shape[1], 2*HIDDEN_SIZE)
            c_out = torch.sum(c_out * cm, dim=0)

            # q_out: (sent_len, batch_size, num_directions*HIDDEN_DIM)
            q_hid, q_out = question_model(torch.transpose(q, 0, 1))
            qm = torch.transpose(qm, 0, 1).unsqueeze(2)
            qm = qm.expand(qm.shape[0], qm.shape[1], 2*HIDDEN_SIZE)
            q_out = torch.sum(q_out * qm, dim=0)

            predictions = relevance_model(torch.cat((c_out, q_out), 1)).squeeze(1)
            predictions = torch.nn.functional.sigmoid(predictions)

            l = torch.FloatTensor([float(lab) for lab in l])
            if USE_CUDA:
                l = l.cuda()
            loss = criterion(predictions, l)
            acc = binary_accuracy(predictions, l)
            epoch_loss += loss.item()
            epoch_acc += acc
            num_batches += 1
        
    return epoch_loss / num_batches, epoch_acc / num_batches

