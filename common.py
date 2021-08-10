asp_dict = {i: idx for idx, i in enumerate(['外观', '舒适性', '操控', '动力', '内饰', '空间'])}
idx2asp = {i: idx for idx, i in asp_dict.items()}
max_token_len = 260
tag_dict = {i: idx for idx, i in enumerate(['O', 'B', 'I', 'E', '<pad>'])}
idx2tag = {i: idx for idx, i in tag_dict.items()}
