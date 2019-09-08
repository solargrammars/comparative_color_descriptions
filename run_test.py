import time
import torch
import pickle
import ipdb
import os
import argparse
import json

from os.path import join
from collections import namedtuple
from datetime import datetime

from settings import OUTPUT_PATH
from model_utils import BatchIterator, ModelBatch 
from model import CompModel, CompLoss
from data import TupleGenerator

parser = argparse.ArgumentParser(description="Color Comparatives")
parser.add_argument("--model_path", default="")
parser.add_argument("--batch_size", type=int, default=128)
test_categories = ["seen_pairings", 
        "unseen_pairings",
        "unseen_ref",
        "unseen_comp",
        "fully_unseen"]

parser.add_argument("--test_category",
        default="seen_pairings", 
        choices=test_categories)
args = parser.parse_args()

ColorTuple = namedtuple("ColorTuple", ["ref", "w", "tgt"]) 
device = torch.device("cuda")

model = CompModel()
model.load_state_dict(torch.load(args.model_path))
model.to(device)


with open(join(OUTPUT_PATH,"w2v_sample.pickle"), "rb") as f:
    w2v_sample = pickle.load(f)

print(str(datetime.now()), "Loading original tuples")
tuple_generator = TupleGenerator()
_, _, test_data = tuple_generator.get_vectorized_tuples()

print(str(datetime.now()), "Generating ColorTuples" )
test_tuples  = [ColorTuple(r, w, t) for r,w,t in test_data[args.test_category]]
test_batches = BatchIterator(test_tuples, args.batch_size, ModelBatch, w2v_sample)

loss_function = CompLoss()


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            batch.to_torch_(device)
            outputs = model(batch.examples_w, batch.examples_ref)
            loss = criterion(outputs, batch.examples_ref, batch.examples_tgt)
            loss = loss.mean()
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

test_loss = evaluate(model,test_batches,loss_function)

print(f'| Test loss: {test_loss:.3f}')
