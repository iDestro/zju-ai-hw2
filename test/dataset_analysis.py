import os
import sys
path = "../dataset/cooked"
filenames = os.listdir(path)
max_length = 0
min_length = sys.maxsize
avg_length = 0
res = {}
less_than = 0
tot_cnt = 0
for filename in filenames:
    res[filename] = {}
    cnt = 0
    sub_max_length = 0
    sub_min_length = sys.maxsize
    sum_length = 0
    with open(path+os.sep+filename, encoding='utf-8') as f:
        for line in f.readlines():
            cur_length = len(line)
            tot_cnt += 1
            if cur_length <= 200:
                less_than += 1
            sum_length += cur_length
            cnt += 1
            sub_max_length = max(sub_max_length, cur_length)
            sub_min_length = min(sub_min_length, cur_length)
    res[filename]["sub_max_length"] = sub_max_length
    res[filename]["sub_min_length"] = sub_min_length
    res[filename]["sub_avg_length"] = sum_length / cnt
    max_length = max(sub_max_length, max_length)
    min_length = min(sub_min_length, min_length)
    avg_length += (sum_length / cnt / len(filenames))

for key, value in res.items():
    print(key)
    print(value)
print(min_length, max_length, avg_length)
print(less_than)
print(tot_cnt)
print(less_than / tot_cnt)
