from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import os.path
import random


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(SCRIPT_PATH, "models", "vanila_phrases")
models_paths = [
    os.path.join(MODELS_DIR, "vanila_phrases_gen_v2.2"),
    os.path.join(MODELS_DIR, "vanila_phrases_gen_v2.2.2"),
]

TRAIN_TEXT_PATH = os.path.join(MODELS_DIR, "vanila_phrases_gen_v2.2.2")

TRAIN_TEXT_FILE_PATH = os.path.join(MODELS_DIR, "train_text.txt")

with open(TRAIN_TEXT_FILE_PATH, encoding="utf-8") as text_file:
    text_sample = text_file.readlines()
text_sample = " ".join(text_sample)


def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)

    sorted_chars = [char for char, _ in char_counts]
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])

    return sequence, char_to_idx, idx_to_char


sequence, char_to_idx, idx_to_char = text_to_seq(text_sample)

SEQ_LEN = 256
BATCH_SIZE = 16


def get_batch(sequence):
    trains = []
    targets = []
    for _ in range(BATCH_SIZE):
        batch_start = np.random.randint(0, len(sequence) - SEQ_LEN)
        chunk = sequence[batch_start : batch_start + SEQ_LEN]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)


def evaluate(
    model, char_to_idx, idx_to_char, start_text=" ", prediction_len=200, temp=0.3
):
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)

    inp = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char

    return predicted_text


class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (
            torch.zeros(
                self.n_layers, batch_size, self.hidden_size, requires_grad=True
            ).to(device),
            torch.zeros(
                self.n_layers, batch_size, self.hidden_size, requires_grad=True
            ).to(device),
        )


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

models = []

for model_path in models_paths:
    model = TextRNN(
        input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2
    )
    model.to(device)

    model.load_state_dict(torch.load(models_paths[1]))

    model.eval()

    models.append(model)


def gen_phrase():
    generated = evaluate(
        random.choice(models),
        char_to_idx,
        idx_to_char,
        temp=0.3,
        prediction_len=1000,
        start_text=". ",
    )

    return generated


if __name__ == "__main__":
    print(gen_phrase())
