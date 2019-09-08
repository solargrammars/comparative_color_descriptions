import numpy as np
import torch
#import ipdb

class BatchIterator(object):

    def __init__(self, 
            examples,
            batch_size,
            batch_builder,
            w2v,
            shuffle=False,
            max_len=None):

        self.max_len = max_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.examples = examples
        self.num_batches = ( len( self.examples) + batch_size -1  ) // batch_size
        self.batch_builder = batch_builder
        self.w2v = w2v

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        examples_slice = []
        for i, example in enumerate(self.examples,  1):
            examples_slice.append(example)
            if i > 0 and i % (self.batch_size) == 0:
                yield self.batch_builder(examples_slice, self.w2v, max_len= self.max_len)
                examples_slice = []

        if examples_slice:
            yield self.batch_builder(examples_slice, self.w2v, max_len=self.max_len)

class ModelBatch(object):

    def __init__(self, examples, w2v, max_len=None):

        examples_w, examples_ref, examples_tgt = [], [], []

        for color_tuple in examples:
            examples_w.append(np.hstack((w2v[color_tuple.w[0]], w2v[color_tuple.w[1]] )))
            examples_ref.append(color_tuple.ref)
            examples_tgt.append(color_tuple.tgt)

        self.examples_w   = np.asarray(examples_w).astype(np.float32)
        self.examples_ref = np.asarray(examples_ref).astype(np.float32)
        self.examples_tgt = np.asarray(examples_tgt).astype(np.float32)
        
    def to_torch_(self, device):
        self.examples_w   = torch.from_numpy(self.examples_w).to(device)
        self.examples_ref = torch.from_numpy(self.examples_ref).to(device)
        self.examples_tgt = torch.from_numpy(self.examples_tgt).to(device)
