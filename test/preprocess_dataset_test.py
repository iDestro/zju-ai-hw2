import json
import os
import pickle

tokenizer = json.load(open('../bert-base-chinese/tokenizer.json', encoding='utf-8'))
vacab = tokenizer['model']['vocab']


invalid_token = []
path = "../dataset/raw"
output_path = "../dataset/cooked"
filenames = os.listdir(path)
for filename in filenames:
    with open(path+os.sep+filename, encoding='utf-8') as f:
        for line in f.readlines():
            for ch in line[:-1]:
                if ch not in vacab:
                    invalid_token.append(ch)

invalid_token_set = set(invalid_token)
print(invalid_token_set)

for filename in filenames:
    with open(path+os.sep+filename, encoding='utf-8') as f:
        output_file = open(output_path+os.sep+filename, 'w', encoding='utf-8')
        for line in f.readlines():
            valid_token = []
            for ch in line[:-1]:
                if ch not in invalid_token_set:
                    valid_token.append(ch)
            output_file.write("".join(valid_token) + '\n')
        output_file.close()

