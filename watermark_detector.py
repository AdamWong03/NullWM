# green token ratio detection placeholder
def detect_green_ratio(text, green_vocab):
    words = text.lower().split()
    if not words:
        return 0
    green_count = sum(1 for w in words if w in green_vocab)
    return green_count / len(words)
