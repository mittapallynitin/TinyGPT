import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        # Xavier initialization for weights
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        # Xavier initialization for embedding weights
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.LayerNorm):
        # Layer normalization initialization
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ModelConfig:
    def __init__(self, embed_dim, num_heads, num_layers, max_seq, vocab_size, expander):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq = max_seq
        self.vocab_size = vocab_size
        self.expander = expander
    
    def __repr__(self):
        return (f"ModelConfig(embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
                f"expander={self.expander}, num_layers={self.num_layers}, "
                f"max_seq={self.max_seq}, vocab_size={self.vocab_size})")


class TinyDecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        embed_dim = config.embed_dim
        num_heads = config.num_heads
        expander = config.expander

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expander),
            nn.ReLU(),
            nn.Linear(embed_dim * expander, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x, causal_mask=None):
        _, _, C = x.size()

        # Scale queries and keys to avoid large dot products
        x = x * (1.0 / C**0.5) 
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x


class TinyDecoderModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        embed_dim = config.embed_dim
        num_layers = config.num_layers
        self.max_seq = config.max_seq
        vocab_size = config.vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(self.max_seq, embed_dim))
        self.decoder_blocks = nn.ModuleList([TinyDecoderBlock(config) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        # Precomputed causal mask (registered buffer so it moves with model across devices)
        # Directly create the causal mask with -inf for masked positions and 0 for allowed ones
        causal_mask = torch.tril(torch.ones(self.max_seq, self.max_seq)).bool()
        causal_mask = causal_mask.to(dtype=torch.float32)  # Convert to float32
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))  # Mask False values with -inf
        causal_mask = causal_mask.masked_fill(causal_mask == 1, 0)  # Set True values to 0
        
        # Register the mask as a buffer
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        B, T = x.size()
        x = self.embedding(x) + self.positional_encoding[:T]  # (B, T, C)
        causal_mask = self.causal_mask[:T, :T]  # (T, T), shared across batch

        for block in self.decoder_blocks:
            x = block(x, causal_mask)

        return self.fc_out(x)
    
    def generate_tokens(self, initial_input, max_length=50, temperature=1.0):
        # initial_input: (B, T), batch size and sequence length
        # max_length: max number of tokens to generate
        # temperature: controls randomness of predictions (higher -> more random)

        device = next(self.parameters()).device
        generated = initial_input.to(device)  # Copy input to the same device as model
        out_tokens = generated.clone()
        if generated.size(1) >= self.max_seq:
            generated = generated[:, -self.max_seq:]
        B, T = generated.size()

        for _ in range(max_length):
            logits = self(generated)  # (B, T, vocab_size)

            # Get the last token's logits (T-1)
            last_token_logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature scaling
            last_token_logits /= temperature

            # Sample from the distribution
            probs = F.softmax(last_token_logits, dim=-1)  # (B, vocab_size)
            next_token = torch.multinomial(probs, 1)  # (B, 1)

            # Append the predicted token to the sequence
            generated = torch.cat((generated, next_token), dim=1)  # (B, T+1)
            out_tokens = torch.cat((out_tokens, next_token), dim=1)
            if generated.size(1) >= self.max_seq:
                generated = generated[:, -self.max_seq:]  # Keep only the last max_seq tokens

        return out_tokens