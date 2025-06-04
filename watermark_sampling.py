# apply green bias to logits placeholder
def apply_watermark_bias(logits, tokenizer, green_vocab, gamma=2.0):
    token_ids = list(range(logits.shape[1]))
    green_ids = [i for i in token_ids if tokenizer.decode([i]).strip() in green_vocab]
    logits[:, green_ids] += gamma
    return logits
