from data_analyse_util import load_data

train = load_data('./data/train.json')

print(train[350])

def split(segment):
    seg_list = []
    print(len(segment))
    if len(segment) == 1:
        return (0,1)
    start = 0
    for i, se in enumerate(segment):
        if i == 0:
            continue
        if se == 1:
            seg_list.append((start, i))
            start = i
        if i == len(segment)-1:
            seg_list.append((start, len(segment)))
    return seg_list

print(split([1,1]))