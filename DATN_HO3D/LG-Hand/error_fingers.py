import matplotlib.pyplot as plt
import numpy as np
import os

fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

finger_case1 = [13.381775, 15.471998, 13.571225, 13.562165, 15.701019]
finger_case2 = [16.667421, 19.065861, 17.266753, 16.779076, 17.736797]
finger_case3 = [16.799643, 19.532959, 17.763221, 17.320181, 18.229351]

import numpy as np

plt.figure(figsize=(20, 15))
barWidth = 0.25
br1 = np.arange(len(fingers))
br2 = [x + barWidth for x in br1]
br3 = [x + 2*barWidth for x in br1]

plt.bar(br1, finger_case1, color="blue", width=barWidth, label="LG-Hand")
plt.bar(br2, finger_case2, color="red", width=barWidth, label="SST-GCN")
plt.bar(br3, finger_case3, color="green", width=barWidth, label="ST-GCN")
plt.xticks([r + barWidth for r in range(len(finger_case3))], fingers, fontsize=40)
plt.ylim(0, 25)
plt.yticks(fontsize=40)
plt.xlabel("Finger", fontweight="bold", fontsize=45)
plt.ylabel("MPJPE(mm)", fontweight='bold', fontsize=45)

plt.legend(fontsize=40)

plt.savefig("ErrorfingersHO3D.png")