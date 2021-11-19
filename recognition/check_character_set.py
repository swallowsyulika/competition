import os
from utils import file_to_list

test_set = "/home/tingyu/projects/competition/utils/generated/output2"

test_set_characters = [x for x in os.listdir(test_set)]



ch_4808 = file_to_list("characters_4808.txt")
ch_2312 = file_to_list("gb_2312_l1.txt")
train_extend = file_to_list("train_extend.txt")

total_set = ch_4808 + train_extend# + ch_2312

missing_list = []
for ch in test_set_characters:
    if ch not in total_set:
        missing_list.append(ch)

print(missing_list)
print(len(missing_list))