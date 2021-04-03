"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset
Usage:
python training_nli.py
OR
python training_nli.py pretrained_transformer_model_name
"""
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
from cnn_pools import CNNSmall, CNNLarge

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it
nli_dataset_path = 'datasets/AllNLI.tsv.gz'
sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'

if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

ap = argparse.ArgumentParser("arguments for bert-nli training")
ap.add_argument('--batch_size',type=int,default=16,help='batch size')
ap.add_argument('--epoch_num',type=int,default=1,help='epoch num')
ap.add_argument('--bert_type',type=str,default='bert-base-uncased',help='transformer (bert) pre-trained model you want to use', choices=['bert-base-uncased','bert-large','albert-base-v2','albert-large-v2'])
ap.add_argument('--pool_type', type=str, default='average', choices=('average', 'cnn-small', 'cnn-large'))
ap.add_argument('--num_layers',type=int,default=12,help='No. of encoder layers for BERT')
ap.add_argument('--log_interval',type=int,default=100,help='No. of steps to log the evaluation')
ap.add_argument('--output_dir', type=str, default='temp')

args = ap.parse_args()

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = args.bert_type

# Read the dataset
train_batch_size = args.batch_size


model_save_path = 'output/' + args.output_dir + '-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, num_layers=args.num_layers)

if args.pool_type == 'average':
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)
elif args.pool_type == 'cnn-small':
    pooling_model = CNNSmall()
elif args.pool_type == 'cnn-large':
    pooling_model = CNNLarge()


model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
train_samples = []
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'train':
            label_id = label2int[row['label']]
            train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))

acc_samples = []
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            label_id = label2int[row['label']]
            acc_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label_id))


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
softmax_train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

acc_dataloader = DataLoader(acc_samples, shuffle=True, batch_size=train_batch_size)

print("sent embed:", model.get_sentence_embedding_dimension())

train_loss = losses.BatchSemiHardTripletLoss(model=model)
softmax_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))


#Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')
acc_evaluator = LabelAccuracyEvaluator(dataloader=acc_dataloader, softmax_model=softmax_loss)

# Configure the training
num_epochs = args.epoch_num

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss), (softmax_train_dataloader, softmax_loss)],
          evaluator=acc_evaluator,
          epochs=num_epochs,
          evaluation_steps=args.log_interval,
          warmup_steps=warmup_steps,
          output_path=model_save_path
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

# model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)

acc_evaluator(model)
