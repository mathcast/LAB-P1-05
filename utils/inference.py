import torch
from utils.tokenizer import tokenize, START_TOKEN, EOS_TOKEN, PAD_TOKEN
from utils.padding import pad_sequence, create_look_ahead_mask

MAX_LEN = 20

def translate(model, sentence):
    model.eval()

    src = tokenize(sentence)
    src = pad_sequence(src, MAX_LEN, PAD_TOKEN)
    src = torch.tensor(src).unsqueeze(0)

    tgt = [START_TOKEN]

    for _ in range(MAX_LEN):
        tgt_tensor = torch.tensor(pad_sequence(tgt, MAX_LEN, PAD_TOKEN)).unsqueeze(0)

        tgt_input = tgt_tensor[:, :len(tgt)]
        mask = create_look_ahead_mask(tgt_input.size(1))

        output = model(src, tgt_input, mask)

        next_token = output.argmax(-1)[0, -1].item()
        tgt.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return tgt