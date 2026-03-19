from datasets import load_dataset


ds = load_dataset("stanfordnlp/sst2")

print("train" , ds['train']['sentence'][:10])
print("test" , ds['test']['sentence'][:10])
