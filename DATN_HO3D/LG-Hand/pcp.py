import matplotlib.pyplot as plt

def readResult(file):
    model = open(file, "r").read().split(", ")
    model = [100*float(x)/13234 for x in model]
    return model

lghand = readResult("resultLGHand.txt")
sstgcn = readResult("resultSSTGCN.txt")
stgcn = readResult("resultSTGCN.txt")

thresholds = [i for i in range(0, len(lghand))]

plt.figure(figsize=(11,7))
plt.plot(thresholds, lghand, "b", label="LG-Hand", linewidth=2)
plt.plot(thresholds, sstgcn, "r:", label="SST-GCN", linewidth=3)
plt.plot(thresholds, stgcn, "g--", label="ST-GCN", linewidth=2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("mm Threshold", fontweight="bold", fontsize=20)
plt.ylabel("Percentage of Correct Predictions", fontweight="bold", fontsize=20)
plt.grid()
plt.legend(fontsize=20)
plt.savefig("compareThresholds.png")
