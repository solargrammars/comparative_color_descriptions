import pickle
import numpy as np
from os import listdir
from datetime import datetime
from os.path import isfile, join, splitext
from gensim.models.keyedvectors import KeyedVectors

from settings import DATA_PATH, GOOGLE_W2V, OUTPUT_PATH
from data_maker import data_maker, clean_data_map, PAD_WORD

class TupleGenerator:
    
    def __init__(self):
        
        print(str(datetime.now()), "Loading comparative data" )
        self.data_map, self.label2word, self.comp_dict, self.vocab = data_maker(
            comp_file=join(DATA_PATH, "comparatives.txt"),
            quant_file=join(DATA_PATH,"quantifiers.txt"),
            file_list=join(DATA_PATH, "words_to_labels.txt")
            )
        
        self.w2v_sample = load_w2v_sample()

        self.train_files, self.dev_files, self.test_files = get_data_files()

        train_labels = [i.split(".")[0] for i in self.train_files]
        dev_labels   = [i.split(".")[0] for i in self.dev_files]
        test_labels  = [i.split(".")[0] for i in self.test_files]

        train_data_map = clean_data_map(self.data_map, train_labels)
        dev_data_map   = clean_data_map(self.data_map, dev_labels)
        test_data_map  = clean_data_map(self.data_map, test_labels)
        
        self.train_tuples = self.generate_tuples(train_data_map)
        self.dev_tuples   = self.generate_tuples(dev_data_map)
        self.test_tuples  = self.generate_tuples(test_data_map)
        
        self.test_tuples_categorized = self.divide_test_set()
        
    def generate_tuples(self, data_map):
        tuples = []
        for ref,v in data_map.items():
            if len(v) > 0:
                for adj, tgt in v:
                    comparative = self.comp_dict[adj[0]].split(" ")
                    if all( i in self.w2v_sample for i in comparative):
                        if len(comparative) == 1: 
                            comparative = ["pad", comparative[0]]
                        tuples.append((ref, comparative, tgt))
        return tuples
    
    def save_tuples(self):
        print(str(datetime.now()), "saving")
        with open(join(OUTPUT_PATH, "train_tuples.pickle"), "wb") as f:
            pickle.dump(self.train_tuples, f)

        with open(join(OUTPUT_PATH, "dev_tuples.pickle"), "wb") as f:
            pickle.dump(self.dev_tuples, f)

        with open(join(OUTPUT_PATH, "test_tuples.pickle"), "wb") as f:
            pickle.dump(self.test_tuples, f)
            
    def divide_test_set(self):
 
        seen_pairings   = []
        unseen_pairings = [] 
        unseen_ref      = []
        unseen_comp     = []
        fully_unseen    = []
        
        # auxiliary for seen pairings
        train_ref_comp = [(i[0], i[1]) for i in self.train_tuples] 
        
        # auxiliary for unseen pairings
        train_refs  = [i[0] for i in self.train_tuples] 
        train_comps = [i[1] for i in self.train_tuples]


        for ref, comp, tgt in self.test_tuples:
            
            # seen pairings
            if (ref,comp) in train_ref_comp:
                seen_pairings.append((ref, comp, tgt))
            
            # unseen pairings   
            elif ref in train_refs and comp in train_comps and (ref,comp) not in train_ref_comp:
                unseen_pairings.append((ref, comp, tgt))
        
            # unseen ref
            elif ref not in train_refs and comp in train_comps:
                unseen_ref.append((ref, comp, tgt))
            
            # unseen comp 
            elif ref in train_refs and comp not in train_comps:
                unseen_comp.append((ref, comp, tgt))
                
            # fully unseen
            elif ref not in train_refs and comp not in train_comps:
                fully_unseen.append((ref, comp, tgt))
                
        total = len(seen_pairings) + len(unseen_pairings) + len(unseen_ref) + len(unseen_comp) + len(fully_unseen)
        assert(len(self.test_tuples) == total)
                
        categorized_test_set = {
            "seen_pairings"   : seen_pairings, 
            "unseen_pairings" : unseen_pairings,
            "unseen_ref"      : unseen_ref,
            "unseen_comp"     : unseen_comp, 
            "fully_unseen"    : fully_unseen
        }        
                
        return categorized_test_set
                
    def vectorize_data(self,tuples, data_dict):

        data = []
        #print(tuples)
        for ref,comp,tgt in tuples:

            refs = data_dict[ref]
            tgts = data_dict[tgt]
            avg_tgt = np.average(tgts, 0).astype(int)

            for r in refs:
                data.append((r, comp, avg_tgt))

        return data
    
    def get_vectorized_tuples(self):
        
        print(str(datetime.now()), "Loading XKCD data")
        train_data_dict = get_xkcd_data(self.train_files)
        dev_data_dict   = get_xkcd_data(self.dev_files)
        test_data_dict  = get_xkcd_data(self.test_files)
        
        print(str(datetime.now()), "Vectorizing tuples")
        train = self.vectorize_data(self.train_tuples, train_data_dict)
        dev   = self.vectorize_data(self.dev_tuples, dev_data_dict)
        
        test = {}
        for category, tuples in self.test_tuples_categorized.items():
            test[category] = self.vectorize_data(tuples, test_data_dict)
       
        return train, dev, test
   
    def get_mapped_test_set(self):
        test_data_dict  = get_xkcd_data(self.test_files)
        test = {}
        for category, tuples in self.test_tuples_categorized.items():
            temp_cont = []
            for tup in tuples:
                temp_cont.append((tup, self.vectorize_data([tup], test_data_dict)))

            test[category] = temp_cont 
        return test

def load_w2v_sample():
    with open(join(OUTPUT_PATH,"w2v_sample.pickle"), "rb") as f:
        w2v_sample = pickle.load(f)
    return w2v_sample

def get_w2v_sample():
    """
    Save a pickle file that contains only the vectors
    associated to the vocabulary under study. 
    
    """
    print(str(datetime.now()), "Loading data")
    data_map, label2word, comp_dict, vocab = data_maker( 
        comp_file=join(DATA_PATH, "comparatives.txt"),
        quant_file=join(DATA_PATH, "quantifiers.txt"),
        file_list=join(DATA_PATH, "words_to_labels.txt")
        )
    
    print(str(datetime.now()), "Loading w2v")
    w2v =  load_google_w2v()
    data = {}

    for w  in comp_dict.values():
        if len(w.split(" ")) == 1:
            if w in w2v:
                data[w] = w2v[w]
        else:
            w_ = w.split(" ")[1]
            if w_ in w2v:
                data[w_] = w2v[w_]

    data["pad"] = np.zeros(300)

    print(str(datetime.now()), "saving")
    with open(join(OUTPUT_PATH, "w2v_sample.pickle"), "wb") as f:
        pickle.dump(data, f)

def get_folder_content(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def get_xkcd_data(file_list):
    file_content = {}
    for fi in file_list:
        with open(join(DATA_PATH, "xkcd_colordata",fi), "rb") as f:
            file_content[fi.split(".")[0]] = pickle.load(f)
    return file_content

def get_data_files():
    files = get_folder_content(join(DATA_PATH, "xkcd_colordata"))
    
    train = [ f for f in files if splitext(f)[1] == ".train"]
    dev   = [ f for f in files if splitext(f)[1] == ".dev"]
    test  = [ f for f in files if splitext(f)[1] == ".test"]
    
    assert(len(files) == ( len(train) + len(dev) + len(test)))
    return train, dev, test

def load_google_w2v():
    """
    Load Google w2v pretrained vectors via Gensim.
    Taken from https://stackoverflow.com/a/44694842
    """
    model = KeyedVectors.load_word2vec_format(
        GOOGLE_W2V, 
        binary=True)
    return model
