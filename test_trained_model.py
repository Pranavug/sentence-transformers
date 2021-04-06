import sys
sys.path.append('../')
sys.path.append('../apex')

import torch
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import argparse

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import argparse
from CNNLarge import CNNLarge
from CNNSmall import CNNSmall
from scipy.stats import pearsonr, spearmanr

from bert_nli import BertNLIModel
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances


def evaluate(model, test_data, checkpoint, mute=False, test_bs=10):
    model.eval()
    sent_pairs = [test_data[i].get_texts() for i in range(len(test_data))]
    all_labels = [test_data[i].get_label() for i in range(len(test_data))]
    with torch.no_grad():
        _, probs = model(sent_pairs,checkpoint,bs=test_bs)
    all_predict = [np.argmax(pp) for pp in probs]
    assert len(all_predict) == len(all_labels)

    acc = len([i for i in range(len(all_labels)) if all_predict[i]==all_labels[i]])*1./len(all_labels)
    prf = precision_recall_fscore_support(all_labels, all_predict, average=None, labels=[0,1,2])

    if not mute:
        print('==>acc<==', acc)
        print('label meanings: 0: contradiction, 1: entail, 2: neutral')
        print('==>precision-recall-f1<==\n', prf)

    return acc


def evaluate_sbert(model, batch_size=16):
    sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
                test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

    sentences1 = []
    sentences2 = []
    scores = []

    examples = test_samples

    for example in examples:
        sentences1.append((example.texts[0], 'none'))
        sentences2.append((example.texts[1], 'none'))
        scores.append(example.label)

    _, embeddings1 = model.forward(sentences1, checkpoint=False)
    _, embeddings2 = model.forward(sentences2, checkpoint=False)
    labels = scores

    cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
    manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
    dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]


    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
    eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

    eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
    eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

    eval_pearson_dot, _ = pearsonr(labels, dot_products)
    eval_spearman_dot, _ = spearmanr(labels, dot_products)

    print("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_cosine, eval_spearman_cosine))
    print("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_manhattan, eval_spearman_manhattan))
    print("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_euclidean, eval_spearman_euclidean))
    print("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
        eval_pearson_dot, eval_spearman_dot))


def parse_args():
    ap = argparse.ArgumentParser("arguments for bert-nli evaluation")
    ap.add_argument('-b','--batch_size',type=int,default=32,help='batch size')
    ap.add_argument('-g','--gpu',type=int,default=1,help='run the model on gpu (1) or not (0)')
    ap.add_argument('-cp','--checkpoint',type=int,default=0,help='run the model with checkpointing (1) or not (0)')
    ap.add_argument('-tm','--trained_model',type=str,default='default',help='path to the trained model you want to test; if set as "default", it will find in output xx.state_dict, where xx is the bert-type you specified')
    ap.add_argument('-bt','--bert_type',type=str,default='bert-base',help='model you want to test; make sure this is consistent with your trained model')
    ap.add_argument('--hans',type=int,default=0,help='use hans dataset (1) or not (0)')

    args = ap.parse_args()
    return args.batch_size, args.gpu, args.trained_model, args.checkpoint, args.bert_type, args.hans

if __name__ == '__main__':
    batch_size, gpu, mpath, checkpoint, bert_type, hans = parse_args()

    if mpath == 'default': mpath = '../bert_nli/output/{}.state_dict'.format(bert_type)
    gpu = bool(gpu)
    hans = bool(hans)
    checkpoint = bool(checkpoint)

    print('=====Arguments=====')
    print('bert type:\t{}'.format(bert_type))
    print('trained model path:\t{}'.format(mpath))
    print('gpu:\t{}'.format(gpu))
    print('checkpoint:\t{}'.format(checkpoint))
    print('batch size:\t{}'.format(batch_size))
    print('hans data:\t{}'.format(hans))

    model = BertNLIModel(model_path=mpath,batch_size=batch_size,bert_type=bert_type)
    evaluate_sbert(model, batch_size=batch_size)



