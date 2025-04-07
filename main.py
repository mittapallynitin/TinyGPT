# main.py
import argparse
import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_setup import CharDataLoader, CharDataset, download_shakespeare
from tiny_decoder import ModelConfig, TinyDecoderModel, init_weights
from tokenizer import CharTokenizer

# ======== Config ========

model_path = "./data/tinyshakespeare.txt"

def main(
    max_seq,
    batch_size,
    epochs,
    generate_new_model
):
    
    # ======== Device-Aware Code ========
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ======== Prepare Dataset ========
    raw_text = download_shakespeare().lower()
    tokenizer = CharTokenizer(raw_text, max_seq=max_seq)

    dataset = CharDataset(raw_text, max_seq)
    dataloader = CharDataLoader(dataset, tokenizer=tokenizer, batch_size=batch_size, shuffle=True)
    print("Dataset size: ", len(dataset))
    print("Vocabulary size: ", tokenizer.vocab_size)
    print("Max sequence length: ", max_seq)
    # # ======== Init Model ========
    model_config = ModelConfig(
        embed_dim = 32,
        num_heads = 4,
        num_layers = 2,
        max_seq = max_seq,
        vocab_size = tokenizer.vocab_size,
        expander = 2,
    )
    if not os.path.exists(model_path) or generate_new_model:
        model = TinyDecoderModel(model_config).to(device)
        model.apply(init_weights)
    else:
        model = TinyDecoderModel(model_config).to(device)
        model.load_state_dict(torch.load("tiny_decoder.pth", weights_only=True))
        print("Model loaded from tiny_decoder.pth")

    model.eval()
    with torch.inference_mode():
        input = "the sun is shining"
        print("Input text: ", input)
        input = tokenizer.encode(input)[0]
        input = torch.tensor(input).unsqueeze(0).to(device)
        tokens = model.generate_tokens(input, max_length=100)
        decoded = tokenizer.decode(tokens[0].cpu().numpy())
        print("Generated text: ", decoded)

    learning_rate = 1e-3
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    print("Number of example per batch: ", max_seq* batch_size)
    # # ======== Training Loop ========
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, (xb, yb) in tqdm(enumerate(dataloader)):
            xb, yb = xb.to(device), yb.to(device)
            # Create causal mask (needed for decoder)
            logits = model(xb)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")



    model.eval()
    with torch.inference_mode():
        input = "the sun is shining"
        print("Input text: ", input)
        input = tokenizer.encode(input)[0]
        input = torch.tensor(input).unsqueeze(0).to(device)
        tokens = model.generate_tokens(input, max_length=100)
        decoded = tokenizer.decode(tokens[0].cpu().numpy())
        print("Generated text: ", decoded)


    # # ======== Save Model ========
    torch.save(model.state_dict(), "tiny_decoder.pth")
    # print("Model saved to tiny_decoder.pth")
    # ======== Load Model ======== 
    model = TinyDecoderModel(model_config).to(device)
    model.load_state_dict(torch.load("tiny_decoder.pth", weights_only=True))


if __name__ == "__main__":
    # Run the script
    # parser = argparse.ArgumentParser(description="Model training command line arguments.")

    # # Define named arguments
    # parser.add_argument("--max_seq", type=int, help="max Seq", required=True)
    # parser.add_argument("--batch_size", type=int, help="Batch Size", required=True)
    # parser.add_argument("--epochs", type=int, help="epochs", required=True)
    # parser.add_argument("--new_model", action='store_true', help="epochs", required=False)

    # # Parse the arguments
    # args = parser.parse_args()
    # # Access the arguments
    # max_seq = args.max_seq
    # batch_size = args.batch_size
    # epochs = args.epochs
    # generate_new_model = args.new_model
    # print("Arguments received:")
    # print(f"Max Seq: {max_seq}, Batch Size: {batch_size}, Epochs: {epochs}, New Model: {generate_new_model}")
    # main(max_seq, batch_size, epochs, generate_new_model)
    # ======== Load Model ======== 
    raw_text = download_shakespeare().lower()
    tokenizer = CharTokenizer(raw_text, max_seq=32)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model_config = ModelConfig(
        embed_dim = 32,
        num_heads = 4,
        num_layers = 2,
        max_seq = 32,
        vocab_size = tokenizer.vocab_size,
        expander = 2,
    )
    model = TinyDecoderModel(model_config).to(device)
    model.load_state_dict(torch.load("./checkpoints/tiny_decoder.pth", weights_only=True))
    from torchinfo import summary
    print(summary(model))
    model.eval()
    with torch.inference_mode():
        input = "the sun is shining"
        print("Input text: ", input)
        input = tokenizer.encode(input)[0]
        input = torch.tensor(input).unsqueeze(0).to(device)
        tokens = model.generate_tokens(input, max_length=1000)
        decoded = tokenizer.decode(tokens[0].cpu().numpy())
        print("Generated text: ", decoded)
        print("-" * 50)