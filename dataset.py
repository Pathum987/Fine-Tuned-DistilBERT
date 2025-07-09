#step2: Import necessary libraries and load dataset

from tensorflow.keras.datasets import imdb
import numpy as np

# Load IMDb dataset with top 10,000 words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Function to get balanced subset
def get_balanced_subset(X, y, num_samples_per_class):
    pos_idx = np.where(y == 1)[0][:num_samples_per_class]
    neg_idx = np.where(y == 0)[0][:num_samples_per_class]
    selected_idx = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(selected_idx)
    return X[selected_idx], y[selected_idx]

# Get 2500 positive and 2500 negative samples for training
X_train_sub, y_train_sub = get_balanced_subset(X_train, y_train, 2500)

#step3: Decode reviews for better readability

# Get word index mapping
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    # IMDb word indices offset by 3
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Decode all reviews in subset
X_train_sub_text = [decode_review(review) for review in X_train_sub]

#step4: Tokenize the text data using Hugging Face's Transformers library

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_inputs = tokenizer(
    X_train_sub_text,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

#step 5: Create a custom dataset class for PyTorch

import torch
from torch.utils.data import Dataset

class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(tokenized_inputs, y_train_sub)
