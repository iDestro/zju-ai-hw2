# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextCNN'
        self.class_list = [x.strip() for x in open('./dataset/class.txt', encoding='utf-8').readlines()]      # 类别名单
        self.save_path = './saved_dict/' + self.model_name + '.pth'        # 模型训练结果
        self.log_path = './log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.3                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 1e-3                                       # 学习率
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                    # 卷积核数量(channels数)
        self.embed = 768


'''Convolutional Neural Networks for Sentence Classification'''


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    model = Net(config=Config())
    print(model)