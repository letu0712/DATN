import matplotlib.pyplot as plt
import numpy as np
import os

fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

finger_case1 = [16.194583, 16.939363, 16.699062, 16.863543, 18.197918]
finger_case2 = [19.098868, 20.047249, 18.637335, 19.560453, 20.673296]
finger_case3 = [19.550547, 21.072115, 18.732210, 19.712662, 20.678074]

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
plt.ylim(0, 30)
plt.yticks(fontsize=40)
plt.xlabel("Finger", fontweight="bold", fontsize=45)
plt.ylabel("MPJPE(mm)", fontweight='bold', fontsize=45)

plt.legend(fontsize=40)

plt.savefig("Error_fingers.png")