import matplotlib.pyplot as plt
import numpy as np
import os

joints = ["Wrist", "MCP", "PIP", "DIP", "TIP"]

joint_case1 = [17.132560, 17.073232, 15.280518, 16.390623, 19.171203]
joint_case2 = [20.995678, 22.070342, 18.374265, 18.459098, 19.510058]
joint_case3 = [22.562744, 22.308628, 18.662002, 19.068727, 19.757130]

import numpy as np

plt.figure(figsize=(20, 15))
barWidth = 0.25
br1 = np.arange(len(joints))
br2 = [x + barWidth for x in br1]
br3 = [x + 2*barWidth for x in br1]

plt.bar(br1, joint_case1, color="blue", width=barWidth, label="LG-Hand")
plt.bar(br2, joint_case2, color="red", width=barWidth, label="SST-GCN")
plt.bar(br3, joint_case3, color="green", width=barWidth, label="ST-GCN")
plt.xticks([r + barWidth for r in range(len(joint_case3))], joints, fontsize=40)
plt.ylim(0, 30)
plt.yticks(fontsize=40)
plt.xlabel("Type of joints", fontweight="bold", fontsize=45)
plt.ylabel("MPJPE(mm)", fontweight='bold', fontsize=45)

plt.legend(fontsize=40)

plt.savefig("Error_joints.png")