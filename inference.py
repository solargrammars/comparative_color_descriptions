import sys
import torch
import pickle
import ipdb
import numpy as np
from os.path import join
from collections import namedtuple

from model import CompModel, CompLoss
from settings import OUTPUT_PATH
from model_utils import BatchIterator, ModelBatch
from data import TupleGenerator

def get_results(model_path, test_category="seen_pairings"):

    with open(join(OUTPUT_PATH,"w2v_sample.pickle"), "rb") as f:
        w2v_sample = pickle.load(f)
 
    ColorTuple = namedtuple("ColorTuple", ["ref", "w", "tgt"])
    tuple_generator = TupleGenerator()
    loss_function = CompLoss()
    
    device = torch.device("cuda")
    model = CompModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    mapped_test_set = tuple_generator.get_mapped_test_set()

    
    
    result = []
    for (t,v) in mapped_test_set[test_category]:
        tuples = [ColorTuple(r,w,t) for  r,w,t in v]
        batches = BatchIterator(tuples,128,ModelBatch, w2v_sample)
        
        temp_cont = []
        for batch in batches: 
            batch.to_torch_(device)
            outputs = model(batch.examples_w, batch.examples_ref)
            loss = loss_function(outputs,batch.examples_ref, batch.examples_tgt)
        
            outputs = outputs.cpu().data.numpy()
            loss = loss.cpu().data.numpy()
            
            temp_cont.extend(list(zip(
                batch.examples_ref.cpu().data.numpy(), 
                batch.examples_tgt.cpu().data.numpy(), 
                outputs,
                loss)))


        result.append((t,temp_cont ))
    
    return result
