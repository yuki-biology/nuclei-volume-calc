import json
import os

# 平均値 (これくらい標準で実装してくれ)
def mean(array):
    return sum(array)/ (len(array) or 1)


# JSONの読み込み
def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

# JSONの保存
def save_json(filepath, data, indent=4):
    print("saving:", filepath)
    dirname = os.path.dirname(filepath)
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    with open(filepath, 'w') as f:
        f.write(json.dumps(data, indent=indent))

# 配列の共通部分を抽出
def extract_common_items_from_lists(*lists):
    items = set(lists[0])
    for list in lists:
        items = items & set(list)
    return list(items)

# 辞書の複数キーを抽出
def extract_dict_by_keys(dict, keys):
    common_keys = extract_common_items_from_lists(dict.keys(), keys) 
    result = {}
    for key in common_keys:
        result[key] = dict[key]
    return result

# 配列を検索
def search_array_by_func(array, func):
    for i, v in enumerate(array):
        if func(v):
            return i
    return -1

def search_array_by_value(array, value):
    return search_array_by_func(array, lambda v: v == value)

def search_array_by_value_in_list(array, value_list):
    return search_array_by_func(array, lambda v: v in value_list)

def search_array_by_value_not_in_list(array, value_list):
    return search_array_by_func(array, lambda v: v not in value_list)

def search_array_by_value_and_key(array, key, value):
    return search_array_by_func(array, lambda v: v[key] == value)
