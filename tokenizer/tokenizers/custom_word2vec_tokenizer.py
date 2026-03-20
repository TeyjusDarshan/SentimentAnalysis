import json 
from gensim.utils import simple_preprocess
import torch

# <PAD> token is 0 and <UNK> token is 1

class CustomWord2VecTokenizer:
    def __init__(self, path_to_word_idx_mapping_json, max_len):
        with open(path_to_word_idx_mapping_json, 'r') as f:
            self.word_to_idx = json.load(f)
            self.max_len = max_len
            self.vocab_size = len(self.word_to_idx) + 2
    
    def filter(self, example):
        if len(simple_preprocess(example['sentence'])) == 0:
            return False
        return True
    
    def tokenize(self, batch):
    # 'batch' is a dict, e.g., {"sentence": ["First sentence", "Second sentence", ...]}
        sentences = batch["sentence"]
        
        all_input_ids = []
        all_attention_masks = []

        for sentence in sentences:
            tokens = simple_preprocess(sentence)



            if(len(tokens) == 0):
                print("emptyyyyyyyyyyyyyyyyyyyyy")
            
            # Your existing logic
            input_ids = [self.word_to_idx.get(token, 1) for token in tokens]
            
            # Truncate/Pad
            if len(input_ids) > self.max_len:
                input_ids = input_ids[:self.max_len]
            
            mask = [1] * len(input_ids)
            
            padding_len = self.max_len - len(input_ids)
            if padding_len > 0:
                input_ids += [0] * padding_len
                mask += [0] * padding_len
                
            all_input_ids.append(input_ids)
            all_attention_masks.append(mask)
        
        # Return a dictionary of lists (don't convert to tensors here, 
        # the datasets library handles that better)
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks
        }


