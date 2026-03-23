from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

PAD_TOKEN = tokenizer.pad_token_id
START_TOKEN = tokenizer.cls_token_id
EOS_TOKEN = tokenizer.sep_token_id

def tokenize(sentence):
    return tokenizer.encode(sentence, truncation=True, max_length=20)