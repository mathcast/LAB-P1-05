import torch

def build_vocab(data):
    vocab = set()

    for item in data:
        vocab.update(item["translation"]["en"])
        vocab.update(item["translation"]["pt"])

    vocab = list(vocab)
    stoi = {ch: i for i, ch in enumerate(vocab)}

    return stoi, len(vocab)


def encode(text, stoi, max_len=20):
    tokens = [stoi.get(ch, 0) for ch in text[:max_len]]

    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))

    return torch.tensor(tokens).unsqueeze(0)  # [1, seq_len]


def train(model, data, loss_fn, optimizer, epochs=2):
    stoi, vocab_size = build_vocab(data)

    print(f"📊 Vocab size: {vocab_size}")

    for epoch in range(epochs):
        print(f"\n📚 Epoch {epoch+1}/{epochs}")

        for item in data:
            # 🔥 AGORA TEMOS SRC E TARGET
            src_text = item["translation"]["en"]
            tgt_text = item["translation"]["pt"]

            src = encode(src_text, stoi)
            tgt = encode(tgt_text, stoi)

            output = model(src)

            # 🔥 CORREÇÃO PRINCIPAL: usar target
            loss = loss_fn(
                output.view(-1, output.size(-1)),
                tgt.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Loss: {loss.item():.4f}")