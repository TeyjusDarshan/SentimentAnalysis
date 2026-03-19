from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset



tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()


trainer = BpeTrainer(
    vocab_size=8000, 
    show_progress=True,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

files = ['tokenizer/data/tokeizer_train_ds.txt']

tokenizer.train(files, trainer)

tokenizer.save("ss2_tokenizer.json")