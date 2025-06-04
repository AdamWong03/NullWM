# green/red vocab split placeholder
from transformers import AutoTokenizer
import random

def get_green_red_vocab(tokenizer, ratio=0.5, seed=42):
    vocab = list(tokenizer.get_vocab().keys())
    vocab = [v for v in vocab if v.isalpha() and v.islower() and len(v) > 1]
    random.seed(seed)
    random.shuffle(vocab)
    split = int(len(vocab) * ratio)
    return set(vocab[:split]), set(vocab[split:])
