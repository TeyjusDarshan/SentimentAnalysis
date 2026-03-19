from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from model.SentiNa import SentiNa
import torch.nn as nn
from torch.optim import AdamW
import torch
from alive_progress import alive_bar
from torch import vmap
import torch.optim as optim
from monitors.metric_monitor import MetricsMonitor
from monitors.validation_monitor import ValidationLossMonitor
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train SentiNa Sentiment Analysis Model")

    parser.add_argument("--total_epochs", type=int, default=100)
    parser.add_argument("--num_encoders", type=int, default=5)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--validation_batch_size", type=int, default=64)
    parser.add_argument("--train_stop_patience", type=int, default=10)
    parser.add_argument("--classification_threshold", type=float, default=0.5)
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--lr_scheduler_cooldown", type=int, default=5)
    parser.add_argument("--initial_lr", type=float, default=2e-5)
    return parser.parse_args()

ds = load_dataset("stanfordnlp/sst2")


args = get_args()

#hyperparams
total_epochs = args.total_epochs
num_encoders = args.num_encoders
num_heads = args.num_heads
model_dim = args.model_dim
max_len = args.max_len
train_batch_size = args.train_batch_size
validation_batch_size = args.validation_batch_size
train_stop_patience = args.train_stop_patience
classification_threshold = args.classification_threshold
lr_scheduler_patience = args.lr_scheduler_patience
lr_scheduler_cooldown = args.lr_scheduler_cooldown
initial_lr = args.initial_lr

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

tokenizer = PreTrainedTokenizerFast(tokenizer_file = 'tokenizer/json/ss2_tokenizer.json')
tokenizer.pad_token = "[PAD]"

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation = True, max_length = max_len)

tokenized_ds = ds.map(tokenize_function, batched=True)

print(tokenized_ds)
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

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, 
    mode='min',
    factor=0.1, 
    patience=lr_scheduler_patience, 
    cooldown=lr_scheduler_cooldown
)

total_params = 0
for params in model.parameters():
    total_params += params.numel()

print(f"Total params: {total_params//1000000}M")

validation_monitor = ValidationLossMonitor(patience=train_stop_patience)
metrics_monitor = MetricsMonitor(classification_threshold)



for epoch in range(total_epochs):
    total_loss = 0
    model.train()
    with alive_bar(len(train_dataloader)) as bar:
        for batch in train_dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device) #shape (B, 256)
            attention_mask = batch['attention_mask'].to(device) # (B, 256)
            labels = batch['label'].view(-1).to(device) #(B)

            #forward pass
            outputs = model(input_ids, attention_mask)

            #backprop
            loss = criterion(outputs.view(-1), labels.float())
            total_loss += loss * input_ids.size(0)

            loss.backward()
            optimizer.step()
            bar()

    
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
            loss = criterion(outputs.view(-1), labels.float())
            total_val_loss += loss * input_ids.size(0)
            metrics_monitor.accumulate_metrics(outputs, labels)

        avg_val_loss = total_val_loss/len(tokenized_ds['validation'])

        scheduler.step(avg_val_loss)

        print(f"Validation loss for epoch {epoch}: {avg_val_loss}")
        is_min_loss = validation_monitor.is_min_loss(avg_val_loss)
        is_added = validation_monitor.add_loss(avg_val_loss, epoch + 1)
        metrics_monitor.print_metrics()

        if(is_min_loss):
            print(f"Checkpointing the model. Best possible loss {avg_val_loss}")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, 'checkpoint.pth')


        if(is_added != True):
            print(f"Validation loss is not imporving for more than {train_stop_patience} epochs")
            print("Ending the training")
            break



