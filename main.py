# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn

import data
import model
import os
import os.path as osp

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, help='GPU device id used')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
data_loader = data.Corpus("./data/ptb", args.batch_size, args.max_sql)

        
# WRITE CODE HERE witnin two '#' bar
########################################
# Build LMModel model (bulid your language model here)

########################################

criterion = nn.CrossEntropyLoss()


# WRITE CODE HERE witnin two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.
def evaluate(data_source):
    pass
########################################


# WRITE CODE HERE witnin two '#' bar
########################################
# Train Function
def train():
    pass 
########################################


# Loop over epochs.
for epoch in range(1, args.epochs+1):
    train()
    evaluate()

