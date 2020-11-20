import json
import os

tokenizer = json.load(open('../bert-base-chinese/tokenizer.json', encoding='utf-8'))
vacab = tokenizer['model']['vocab']


invalid_token = []
path = "../dataset/cooked"
filenames = os.listdir(path)
for filename in filenames:
    with open(path+os.sep+filename, encoding='utf-8') as f:
        for line in f.readlines():
            for ch in line[:-1]:
                if ch not in vacab:
                    invalid_token.append(ch)

invalid_token_set = set(invalid_token)
print(invalid_token_set)