import matplotlib.pyplot as plt
import numpy as np
import os

actions = ["open_milk", "open_soda_can", "scratch_sponge", "give_coin", "wash_sponge", "open_letter",
   "unfold_glasses", "open_peanut_butter", "read_letter", "pour_liquid_soap", "pour_wine", "tear_paper", "open_wallet", "put_tea_bag", "high_five", 
  "charge_cell_phone", "handshake", "light_candle", "toast_wine", "scoop_spoon", "flip_sponge", 
  "receive_coin", "close_peanut_butter", "use_flash", "flip_pages", "close_liquid_soap", "close_milk", 
  'squeeze_sponge', 'pour_juice_bottle', 'pour_milk', 'take_letter_from_enveloppe', 'use_calculator', 
  "write", "put_salt", "clean_glasses", "prick", "open_liquid_soap", "open_juice_bottle", 
  "close_juice_bottle", "sprinkle", "give_card", "drink_mug", "stir", "put_sugar", "squeeze_paper", "Average"]

data_raw = open("data_raw.txt", "r")
data_new = []


for row in data_raw:
    digit = ''.join(filter(str.isdigit, row))
    data_new.append(float(digit)/100)


lghand = data_new[:int(len(data_new)/3)]
sstgcn = data_new[int(len(data_new)/3):int(2*len(data_new)/3)]
stgcn = data_new[int(2*len(data_new)/3):]

print(lghand[:46])

barWidth = 0.25
br1 = np.arange(len(actions))
br2 = [x + barWidth for x in br1]
br3 = [x + 2*barWidth for x in br1]


plt.figure(figsize=(40,23))

plt.bar(br1, lghand[:46], color="blue", width=barWidth, label="Case 1: LG-Hand")
plt.bar(br2, sstgcn[:46], color="red", width=barWidth, label="Case 2: SST-GCN")
plt.bar(br3, stgcn[:46], color="green", width=barWidth, label="Case 3: ST-GCN")
plt.xticks([r + barWidth for r in range(len(actions))], actions, rotation=90, fontsize=20)
plt.title("Lỗi ước lượng 3D (mm) theo từng hành động", fontsize=40)
plt.yticks(fontsize=30)
plt.legend(fontsize=40)
plt.savefig("result.png")