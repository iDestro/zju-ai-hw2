import os
from model import Config, Net
from utils import TextDataset
from torch.utils.data import random_split, DataLoader
from train_eval import train

os.system("pip install transformers")

dataset = TextDataset()
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset)-train_size-val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

config = Config()
train_iter = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=0)
val_iter = DataLoader(val_dataset, shuffle=True, batch_size=config.batch_size, num_workers=0)
test_iter = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size, num_workers=0)

model = Net(config).to(config.device)

train(config, model, train_iter, val_iter, test_iter)
