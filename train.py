from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
import re
from transformers import AutoTokenizer
from model.SentiNa import SentiNa
import torch.nn as nn
from torch.optim import AdamW
import torch

ds = load_dataset("stanfordnlp/sst2")
model_id = "google/gemma-3-1b-it"

#hyperparams
total_epochs = 1
num_encoders = 4
num_heads = 4
model_dim = 256
max_len = 512
train_batch_size = 2
validation_batch_size = 2

def get_device():
    """Returns the most powerful available device."""
    if torch.cuda.is_available():
        # NVIDIA GPU
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1/M2/M3/M4)
        device = torch.device("mps")
        print("Device: Apple Silicon (MPS)")
    else:
        # Standard CPU
        device = torch.device("cpu")
        print("Device: CPU (No GPU found)")
    
    return device

# Usage
device = get_device()



'''class Preprocesser:
    def __init__(self, data: DatasetDict):
        self.data = data
        self.stock_name_regex = r"\$[A-Z][A-Z0-9\.]{0,9}"
        self.url_regex = r"\b(https?:\/\/|www\.)\S+\b"
    

    def preprocess(self):
        self.data = self.data.map(self.remove_stock_name)
        self.data = self.data.map(self.remove_url)
        self.data = self.data.map(self.lower_case)
        self.data = self.data.map(self.remove_hyphen)
        self.data = self.data.map(self.strip)
        return self.data

    def strip(self, example):
        example['text'] = example['text'].strip()
        return example
    
    def remove_hyphen(self, example):
        text: str = example['text']
        cleaned = text.replace(" - ", "", 1)
        example['text'] = cleaned
        return example



    def lower_case(self, example):
        example['text'] = example['text'].lower() 
        return example

    def remove_url(self, example):
        example['text'] = re.sub(self.url_regex, "", example['text'])
        return example

    def remove_stock_name(self, example):
        example['text'] = re.sub(self.stock_name_regex, "", example['text'])
        return example

# preprocessor = Preprocesser(ds)
# ds = preprocessor.preprocess()'''


tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation = True, max_length = 256)

tokenized_ds = ds.map(tokenize_function, batched=True)
tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_dataloader = DataLoader(tokenized_ds['train'], shuffle=True, batch_size=train_batch_size)
val_dataloader = DataLoader(tokenized_ds['validation'], shuffle = True, batch_size=validation_batch_size)


model = SentiNa(
    tokenizer.vocab_size,
    num_encoder=num_encoders, 
    num_heads=num_heads, 
    model_dim=model_dim, 
    max_len=max_len
).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

print("size", len(tokenized_ds['validation']))

total_params = 0
for params in model.parameters():
    total_params += params.numel()

print(f"Total params: {total_params//1000000}M")


'''
for epoch in range(total_epochs):
    total_loss = 0
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device) #shape (B, 256)
        attention_mask = batch['attention_mask'].to(device) # (B, 256)
        labels = batch['label'].view(-1).to(device) #(B)

        #forward pass
        outputs = model(input_ids, attention_mask)

        #backprop
        loss = criterion(outputs, labels)
        total_loss += loss * input_ids.size(0)

        loss.backward()

        optimizer.step()
    
    avg_epoch_loss = total_loss / len(tokenized_ds['train'])
    print(f'Train loss for epoch {epoch}: {avg_epoch_loss}')

    #validation loss
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device) #shape (B, 256)
            attention_mask = batch['attention_mask'].to(device) # (B, 256)
            labels = batch['label'].view(-1).to(device) #(B)

            #forward pass
            outputs = model(input_ids, attention_mask)

            #backprop
            loss = criterion(outputs, labels)
            total_val_loss += loss * input_ids.size(0)

        avg_val_loss = total_val_loss/len(tokenized_ds['validation'])
        print(f"Validation loss for epoch {epoch}: {avg_val_loss}")
'''


     
