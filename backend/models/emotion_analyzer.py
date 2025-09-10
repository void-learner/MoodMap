from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Load the dataset
dataset = load_dataset("go_emotion", "simplified")

# Preprocess
