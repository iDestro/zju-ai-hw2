import time
import os
import torch
from datetime import timedelta
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertConfig, BertModel


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def preprocess(path, max_length=200):
    class2index = {value: key for key, value in enumerate(["LX.txt", "MY.txt", "QZS.txt", "WXB.txt", "ZAL.txt"])}
    data = []
    targets = []
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model_config = BertConfig.from_pretrained("bert-base-chinese")
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    bert_model = BertModel.from_pretrained("bert-base-chinese", config=model_config).cuda()
    filenames = os.listdir(path)

    tot_sentences = 0
    for filename in filenames:
        with open(path + os.sep + filename, encoding='utf-8') as f:
            tot_sentences += len(f.readlines())

    with torch.no_grad():
        cnt = 0
        filenames = os.listdir(path)
        for filename in filenames:
            with open(path+os.sep+filename, encoding='utf-8') as f:
                for line in f.readlines():
                    cnt += 1
                    line = line[:-1]
                    input = tokenizer(line, return_tensors="pt")
                    valid_token_cnt = input['input_ids'].numel()-2
                    line = line[:max_length] if valid_token_cnt > max_length else line + "[PAD]" * (max_length - valid_token_cnt)
                    input = tokenizer(line, return_tensors="pt")
                    output = bert_model(input['input_ids'].cuda())
                    data.append(output[0].cpu())
                    targets.append(class2index[filename])
                    if cnt % 10 == 0:
                        print(cnt / tot_sentences * 100)

    data = torch.cat(data, dim=0)
    targets = torch.LongTensor(targets)
    print(data.size())
    print(targets.size())
    torch.save(data, './saved_dict/data.pt')
    torch.save(targets, './saved_dict/targets.pt')
    return data, targets


def sentence2vec(text, max_length=200):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model_config = BertConfig.from_pretrained("bert-base-chinese")
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    bert_model = BertModel.from_pretrained("bert-base-chinese", config=model_config).cuda()
    with torch.no_grad():
        input = tokenizer(text, return_tensors="pt")
        valid_token_cnt = input['input_ids'].numel() - 2
        line = text[:max_length] if valid_token_cnt > max_length else text + "[PAD]" * (max_length - valid_token_cnt)
        input = tokenizer(line, return_tensors="pt")
        output = bert_model(input['input_ids'].cuda())
    return output[0]


class TextDataset(Dataset):
    def __init__(self):
        if os.path.exists("./saved_dict/data.pt") and os.path.exists("./saved_dict/targets.pt"):
            self.data = torch.load("data.pt")
            self.targets = torch.load("targets.pt")
        else:
            self.data, self.targets = preprocess('./dataset/cooked')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


if __name__ == '__main__':
    # preprocess('./dataset/cooked')
    sen = "我听到一声尖叫，感觉到蹄爪戳在了一个富有弹性的东西上。定睛一看，不由怒火中烧。原来，趁着我不在，隔壁那个野杂种——沂蒙山猪刁小三，正舒坦地趴在我的绣榻上睡觉。我的身体顿时痒了起来，我的目光顿时凶了起来。"
    print(sentence2vec(sen).size())