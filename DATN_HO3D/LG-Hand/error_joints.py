import matplotlib.pyplot as plt
import numpy as np
import os

joints = ["Wrist", "MCP", "PIP", "DIP", "TIP"]

joint_case1 = [14.495614, 13.803894, 13.557764, 14.515058, 15.473831]
joint_case2 = [16.215166, 16.416290, 17.637153, 17.642776, 18.316510]
joint_case3 = [16.413963, 16.620010, 18.133548, 18.082690, 18.880037]

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
plt.ylim(0, 23)
plt.yticks(fontsize=40)
plt.xlabel("Type of joints", fontweight="bold", fontsize=45)
plt.ylabel("MPJPE(mm)", fontweight='bold', fontsize=45)

plt.legend(fontsize=40)

plt.savefig("ErrorjointsHO3D.png")