import time
import torch
import pickle
import ipdb
import os
import argparse
import json

from os.path import join
from collections import namedtuple
from tensorboardX import SummaryWriter
from datetime import datetime

from settings import OUTPUT_PATH
from model_utils import BatchIterator, ModelBatch 
from model import CompModel, CompLoss
from data import TupleGenerator

parser = argparse.ArgumentParser(description="Color Comparatives")
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--validate", action="store_true")
args = parser.parse_args()

ColorTuple = namedtuple("ColorTuple", ["ref", "w", "tgt"]) 
device = torch.device("cuda")

with open(join(OUTPUT_PATH,"w2v_sample.pickle"), "rb") as f:
    w2v_sample = pickle.load(f)

print(str(datetime.now()), "Loading original tuples")
tuple_generator = TupleGenerator()
train_data, valid_data, _ = tuple_generator.get_vectorized_tuples()

print(str(datetime.now()), "Generating training set")
train_tuples = [ColorTuple(r, w, t) for r,w,t in train_data]
train_batches = BatchIterator(train_tuples, args.batch_size, ModelBatch, w2v_sample)

if args.validate == True:
    print(str(datetime.now()), "Generating validation set") 
    valid_tuples  = [ColorTuple(r, w, t) for r,w,t in valid_data]
    valid_batches = BatchIterator(valid_tuples, args.batch_size, ModelBatch, w2v_sample)

m = CompModel()
m = m.to(device=device)
print(m)

model_id = str(int(time.time())) 
save_path = os.path.join(OUTPUT_PATH, model_id)
if not os.path.isdir(save_path):
    os.makedirs(save_path)
writer = SummaryWriter(save_path)

print("Model id: ", model_id)

with open(os.path.join(save_path,"params.json"), "w") as f:
    json.dump(vars(args), f)

optimizer = torch.optim.Adam(m.parameters(), lr=args.lr)
loss_function = CompLoss()

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0

    for batch in iterator:
        batch.to_torch_(device)
        optimizer.zero_grad()
        outputs = model(batch.examples_w, batch.examples_ref)
        loss = criterion(outputs, batch.examples_ref, batch.examples_tgt)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

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

print(str(datetime.now()), "Begin training")

if args.validate == True:
    best_valid_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = train(m, train_batches, optimizer, loss_function)
        valid_loss = evaluate(m, valid_batches, loss_function)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(m.state_dict(), join(save_path,"model.pth"))
   
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('valid_loss', valid_loss, epoch)
        print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} |')
else:
    for epoch in range(args.epochs):
        train_loss = train(m, train_batches, optimizer, loss_function)
        writer.add_scalar('train_loss', train_loss, epoch)
        print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} |')
    torch.save(m.state_dict(), join(save_path,"model.pth"))
print(str(datetime.now()), "Finished training, bye bye")
