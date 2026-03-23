import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import load_data
from models.transformer import Transformer
from training.train import train, build_vocab

def main():
    data = load_data()

    # cria vocab antes de instanciar modelo
    stoi, vocab_size = build_vocab(data)

    model = Transformer(vocab_size=vocab_size, d_model=128)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Iniciando treino...")
    train(model, data, loss_fn, optimizer, epochs=2)

if __name__ == "__main__":
    main()