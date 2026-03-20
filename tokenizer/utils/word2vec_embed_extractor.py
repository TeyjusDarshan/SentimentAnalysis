import gensim.downloader as api
import json

from gensim.utils import simple_preprocess
from datasets import load_dataset
import torch.nn as nn
import torch
import numpy as np


ds = load_dataset("stanfordnlp/sst2")


model = api.load("word2vec-google-news-300")

# print(model['cat'])

print(model.key_to_index['Cat'])

words_found = 0
words_not_found = 0

key_to_idx_dict = {}
current_idx = 2

embeddings = []


for sentence in ds['validation']['sentence']:
    tokens = simple_preprocess(sentence)

    for token in tokens:
        if token not in key_to_idx_dict:
            if token in model: 
                embed = model[token]
                key_to_idx_dict[token] = current_idx
                embeddings.append(embed)
                current_idx += 1

for sentence in ds['train']['sentence']:
    tokens = simple_preprocess(sentence)

    for token in tokens:
        if token not in key_to_idx_dict:
            if token in model: 
                embed = model[token]
                key_to_idx_dict[token] = current_idx
                embeddings.append(embed)
                current_idx += 1

for sentence in ds['test']['sentence']:
    tokens = simple_preprocess(sentence)

    for token in tokens:
        if token not in key_to_idx_dict:
            if token in model: 
                embed = model[token]
                key_to_idx_dict[token] = current_idx
                embeddings.append(embed)
                current_idx += 1

embedding_tensor = torch.from_numpy(np.stack(embeddings)).float()
unknown_word_embedding = torch.mean(embedding_tensor, dim = 0)
final_embedding_tensor = torch.cat((unknown_word_embedding.unsqueeze(0), unknown_word_embedding.unsqueeze(0), embedding_tensor)) #0 is padding 1 is <UNK> 



with open("word2vec_tokenizer.json", "w", encoding='utf-8') as f:
    json.dump(key_to_idx_dict, f, indent=4, ensure_ascii=False)

torch.save(final_embedding_tensor, "embedding_weights.pt")



