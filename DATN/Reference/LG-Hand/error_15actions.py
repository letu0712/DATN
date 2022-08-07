import matplotlib.pyplot as plt
import numpy as np
import os


actions = ["wash_sponge",
"unfold_glasses",
"open_peanut_butter",
"read_letter",
"tear_paper",
"put_tea_bag",
"light_candle",
"toast_wine",
"flip_sponge",
"close_peanut_butter",
"use_flash",
"close_milk",
"pour_milk",
"open_juice_bottle",
"squeeze_paper"]

data_raw = open("data_raw_15.txt", "r")
data_new = []


def insert(string, index, str_insert):
    len_str = len(string)
    new_str = string[:index] + str_insert + string[index:]
    return new_str

for data in data_raw.readlines():
    new = ''.join(filter(str.isdigit, data))
    new = float(insert(new, 2, "."))
    data_new.append(new)

print(len(data_new))

num_case = 3
num_item = int(len(data_new)/num_case)

case1 = data_new[:num_item]
case2 = data_new[num_item:num_item*2]
case3 = data_new[num_item*2:]

action_case1 = case1[:len(actions)]           # +1 average
action_case2 = case2[:len(actions)]
action_case3 = case3[:len(actions)]

print(action_case1)
print(action_case2)
print(action_case3)


import numpy as np

plt.figure(figsize=(50, 30))
barWidth = 0.25
br1 = np.arange(len(actions))
br2 = [x + barWidth for x in br1]
br3 = [x + 2*barWidth for x in br1]

plt.bar(br1, action_case1, color="blue", width=barWidth, label="Case 1: LG-Hand")
plt.bar(br2, action_case2, color="red", width=barWidth, label="Case 2: SST-GCN")
plt.bar(br3, action_case3, color="green", width=barWidth, label="Case 3: ST-GCN")
plt.xticks([r + barWidth - 0.5 for r in range(len(actions))], actions, rotation=30, fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Actions", fontweight="bold", fontsize=40)
plt.ylabel("Error(mm)", fontweight='bold', fontsize=40)
plt.title("MPJPE 15 actions", fontsize=60, fontweight="bold")
plt.legend(fontsize=40)

plt.savefig("Error_15actions.png")