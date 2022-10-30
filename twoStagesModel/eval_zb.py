


def eval_acc(valid_datapath, pred_datapath):

    valid_data = {}
    file = open(valid_datapath, encoding='utf-8')
    while 1:
        line = file.readline()
        if not line:
            break
        pre = eval(line)

        # 提取begin
        key_points = pre["key_points"]
        valid_beg = [int(kp["begin"]) for kp in key_points]

        valid_data[pre["url"]] = valid_beg
    file.close()

    pred_data = {}
    file = open(pred_datapath, encoding='utf-8')
    while 1:
        line = file.readline()
        if not line:
            break
        pre = eval(line)

        # 提取begin
        key_points = pre["key_points"]
        pred_beg = [int(kp["begin"]) for kp in key_points]

        pred_data[pre["url"]] = pred_beg
    file.close()

    assert len(pred_data) == len(valid_data)

    def inter(a, b):
        return list(set(a) & set(b))
    #
    precision = 0
    recall = 0
    for url in valid_data.keys():
        v_beg = valid_data[url]
        p_beg = pred_data[url]

        con_beg = inter(v_beg, p_beg)
        if len(con_beg) == 0 :
            continue
        precision += len(con_beg) / len(p_beg)
        recall += len(con_beg) / len(v_beg)

    precision /= len(valid_data)
    recall /= len(valid_data)
    f1 = 2*(precision * recall) / (precision + recall)

    print("Precision: {}; Recall: {}; F1: {}".format(precision, recall, f1))
    return f1










