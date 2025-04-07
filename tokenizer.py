class CharTokenizer:
    def __init__(self, data, special_tokens=None, max_seq=32, padding=True):
        if special_tokens is None:
            special_tokens = ["<pad>"]

        self.data = data
        self.special_tokens = special_tokens
        vocab, stoi, itos = self.build_vocab()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.char_to_index = stoi
        self.index_to_char = itos
        self.max_seq = max_seq
        self.padding = padding
  
        
    def build_vocab(self):
      chars = sorted(list(set(self.data)))
      chars = self.special_tokens + chars
      stoi = {ch: i for i, ch in enumerate(chars)}
      itos = {i: ch for ch, i in stoi.items()}
      return chars, stoi, itos
    
    def encode(self, text) -> list[int]:
        encoded = [self.char_to_index[char] for char in text]
        attention_mask = [1] * len(encoded)
        # Padding the sequence if necessary
        if len(encoded) < self.max_seq and self.padding:
            padding_length = self.max_seq - len(encoded)
            encoded.extend([self.char_to_index["<pad>"]] * padding_length)
            attention_mask.extend([0] * padding_length)
          
        return encoded, attention_mask
    
    def decode(self, encoded) -> str:
        decoded = [self.index_to_char[idx] for idx in encoded if idx != self.char_to_index["<pad>"]]
        return "".join(decoded)
    
    def __call__(self, text):
        return self.encode(text)
