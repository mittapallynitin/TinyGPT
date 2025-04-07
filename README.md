# TinyGPT 🧠🔥

TinyGPT is a lightweight, educational transformer decoder-only model inspired by GPT-like architectures. This project walks through building a transformer decoder block from scratch using PyTorch, training it on synthetic or tokenized text data, and generating text in an autoregressive fashion.

---

## 🏗️ Architecture Overview

TinyGPT is a decoder-only transformer architecture composed of:

- **Embedding Layer**: Converts input token IDs into dense vectors.
- **Positional Encoding**: Learnable positional encodings to help model sequence order.
- **Stack of Decoder Blocks**: Each block contains:
  - Multi-head self-attention (with causal masking)
  - Layer Normalization
  - Feedforward Network (2-layer MLP)
  - Residual Connections
- **Final Linear Layer**: Projects the decoder output to vocabulary size logits for prediction.

### ✨ Attention Mechanism
Each decoder block uses **Multi-head Self-Attention** with a **causal mask**. The causal mask ensures that a token can only attend to itself and previous tokens (not future tokens), maintaining autoregressive generation integrity.

The attention is implemented using PyTorch’s `nn.MultiheadAttention` with the `attn_mask` argument used to apply the lower-triangular causal mask:

```python
causal_mask = torch.tril(torch.ones(max_seq, max_seq)).bool()
```

This is registered as a buffer in the model and applied during the forward pass.

---

## ✍️ Character-Level Tokenizer (CharTokenizer)

This project uses a custom character-level tokenizer, which:
	•	Builds a vocabulary from all unique characters in the dataset.
	•	Converts input strings to sequences of token IDs (integers).
	•	Provides both encode() and decode() methods to convert between text and tokens.
	•	Handles special cases like unknown characters gracefully.

Character-level tokenization ensures simplicity and helps when working with small models or datasets.

Example:
```python
text = "hello"
tokens = tokenizer.encode(text)  # [5, 2, 7, 7, 9]
decoded = tokenizer.decode(tokens)  # "hello"
```

## 🧪 Training Logic

### 🔄 Dataset Creation
Training samples are created from a long stream of text (e.g., tokenized documents or code) by chunking it into fixed-size sequences (e.g., `seq_len = 64`). For each training example:

- The **input** `x` is a slice of tokens: `[t_0, t_1, ..., t_n-1]`
- The **target** `y` is a right-shifted version: `[t_1, t_2, ..., t_n]`

This shifting logic allows the model to learn to predict the next token at each timestep.

### 📉 Loss Computation
The model outputs logits of shape `(B, T, vocab_size)`. The target labels are of shape `(B, T)`. We compute the cross-entropy loss by flattening both:

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

This computes the average negative log likelihood across all tokens in the batch.

---

## 🚀 Generation Logic

The model generates text autoregressively:

1. Start with a prompt or token (e.g., `[BOS]`).
2. Feed into the model to get logits.
3. Sample the next token from the logits.
4. Append the token to the sequence.
5. Truncate the sequence if it exceeds `max_seq` length.
6. Repeat for desired length.

To preserve the original sequence and history, we maintain `out_tokens` as the full list of generated tokens, and `generated` as the truncated sequence passed to the model each time:

```python
for _ in range(max_new_tokens):
    logits = model(generated)
    next_token = sample(logits[:, -1, :])
    out_tokens = torch.cat((out_tokens, next_token), dim=1)
    generated = out_tokens[:, -max_seq:]  # sliding window
```

---

## 🧠 Gotchas & Lessons Learned

- **Causal Mask Shape**: Must be `(T, T)` and broadcastable to the attention layer.
- **NaNs During Training**: Caused by improperly masked attention or high logits. Solved by proper causal masking and scaling.
- **Mask in torch**: When using built in transformer layer the mash should have 0 and -inf not 1 and 0 for seq to attend to.
- **Shifted Targets**: Critical to shift targets to ensure next-token prediction.
- **Training Stability**: Initializing weights using Xavier initialization and applying LayerNorm stabilized early training.
- **Logits Softmax**: During generation, use temperature scaling and softmax before sampling.

---

## 🧾 Future Work

- Add tokenizer + tokenizer training script
- LoRA/PEFT based fine-tuning
- Implement Top-k and Top-p sampling
- Add beam search
- Train on Project Gutenberg corpus

---

## 📊 Results

When trained on small toy datasets, TinyGPT is able to generate coherent sequences that follow training distribution patterns. More improvements can be made by training on richer corpora and increasing model depth. The next steps from this
project is to experiment with the KV Caching methods to speed up the inference. 

---

## 🧑‍💻 Author
**Nitin Mittapally**  
Machine Learning Engineer

---

## 📂 Folder Structure
```
TinyGPT/
├── tiny_decoder.py             # TinyGPT architecture
├── main.py                     # Training loop and data prep
├── data_setup.py               # Text generation logic
├── tokenizer.py                # Tokenizer logic
├── data/
├──── tinyshakespeare.txt       # Raw datasets
├── checkpoints/
├──── tiny_decoder.pth          # Saved weights
└── README.md
```

---

If you found this useful, consider giving this repo a ⭐!

