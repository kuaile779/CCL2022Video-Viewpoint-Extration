
import json
import numpy as np
np.random.seed(100)


def split_train(filename: str):

    id = 0
    file = open(filename, encoding='utf-8')

    total_data = []
    while 1:
        line = file.readline()
        if not line:
            break
        pre = eval(line)
        total_data.append(pre)

        id += 1
    file.close()
    #
    print('get {} examples'.format(id))  # 10585

    # 拆分训练集与验证集
    np.random.shuffle(total_data)
    train_data = total_data[:10000]
    valid_data = total_data[10000:]
    print('get {} train examples'.format(len(train_data)))
    print('get {} valid examples'.format(len(valid_data)))

    for exam in train_data:
        with open('../data/new-train.jsonl', "a+", encoding='utf-8') as outFile:
            outFile.write(json.dumps(exam, ensure_ascii=False) + '\n')
    for exam in valid_data:
        with open('../data/new-valid.jsonl', "a+", encoding='utf-8') as outFile:
            outFile.write(json.dumps(exam, ensure_ascii=False) + '\n')

    return 0

split_train("../data/train.jsonl")

