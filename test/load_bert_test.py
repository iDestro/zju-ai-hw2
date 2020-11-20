from transformers import AutoTokenizer, BertConfig, BertModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model_config = BertConfig.from_pretrained("bert-base-chinese")
model_config.output_hidden_states = True
model_config.output_attentions = True
bert_model = BertModel.from_pretrained("bert-base-chinese", config=model_config)

with torch.no_grad():
    input = tokenizer("我我我", return_tensors="pt")
    print(input)
    # output = bert_model(input['input_ids'].cuda())
    output = bert_model(**input)
print(output[0])