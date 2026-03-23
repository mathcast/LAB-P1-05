from datasets import load_dataset

def load_data():
    dataset = load_dataset("Helsinki-NLP/opus_books", "en-pt")
    dataset = dataset["train"].select(range(1000))

    return dataset