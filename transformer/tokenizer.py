
class Tokenizer:
    def __init__(self, vocab):
        self.words = vocab.words
        self.control_verbs = set(vocab.control_verbs)
        self.delimiter = vocab.delimiter
        self.has_delimiter = vocab.has_delimiter
        self.token_mapping = {word: i for i, word in enumerate(vocab.words)}

    def encode(self, text):
        if self.has_delimiter:
            text = text.replace(self.delimiter, f" {self.delimiter} ")
        else:
            text = text.replace(self.delimiter, "")
        return [self.token_mapping[word] for word in text.split()]

    def decode(self, ids):
        words = [self.words[i] for i in ids]

        if self.has_delimiter:
            return " ".join(words).replace(f" {self.delimiter} ", f"{self.delimiter} ")

        parts = []
        for word in words:
            if word in self.control_verbs and parts:
                parts.append(self.delimiter)
            parts.append(word)
        return " ".join(parts)
