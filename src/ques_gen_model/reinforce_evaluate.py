from .constants import *
from .helper import *
from ques_gen_model.helper import *
import numpy as np
import torch
from torch.autograd import Variable


def evaluate_relevance(context_model, question_model, relevance_model, c, cl, q, ql, args):
    with torch.no_grad():
        context_model.eval()
        question_model.eval()
        relevance_model.eval()
        cm = get_masks(cl, args.max_post_len)
        qm = get_masks(ql, args.max_ques_len)

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

        q_hid, q_out = question_model(torch.transpose(q, 0, 1))
        qm = torch.transpose(qm, 0, 1).unsqueeze(2)
        qm = qm.expand(qm.shape[0], qm.shape[1], 2*HIDDEN_SIZE)
        q_out = torch.sum(q_out * qm, dim=0)

        predictions = relevance_model(torch.cat((c_out, q_out), 1)).squeeze(1)
        predictions = torch.nn.functional.sigmoid(predictions)

    return predictions
