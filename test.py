# import sqlite3
# print(sqlite3.sqlite_version)
# print(sqlite3.version)

# import torch
# print("CUDA Available:", torch.cuda.is_available())
# print("GPU Name:", torch.cuda.get_device_name(0))
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)

from datasets import load_dataset
dataset = load_dataset("go_emotions", "simplified")
print(dataset)