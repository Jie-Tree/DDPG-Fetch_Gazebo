import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime
sns.set(style="white",palette="muted",color_codes=True)
with open("logold.txt") as f:
	a = f.readlines()
reward = []
step = []
learning = True
size = 10
for i in range(len(a)):
	# if a[i][21:26] == 'learn':
	# 	learning = True
	# if a[i][0] == 'S' and learning and i % size == 0:
	# step.append(int(a[i][:9].split(",")[0].split(":")[1]))
	reward_ = a[i].split("=")[2].replace("loss", "")
	print reward_
	print reward_[:-3]
	# if reward_ > -100:
	reward.append(float(reward_[:-3]))

# number = []
# for i in range(1, 21):
# 	number.append(step.count(i)*size)
# 	print("number of "+str(i)+" step:", number[i-1])
# print("total number:", len(step))
# title1 = "grasp success present:{0}%".format(100*(1-number[19]/sum(number)))
# print(title1)

fig, axes = plt.subplots(1, 1)
axes.plot(np.linspace(1, len(reward)*size,len(reward)), reward)
axes.set_xlabel("Episode")
axes.set_ylabel("Reward")
axes.set_title("the reward after pre-training")
now = datetime.datetime.now()
otherStyleTime = now.strftime("%Y--%m--%d-%H:%M:%S")
plt.savefig('reward_img/'+otherStyleTime)
plt.show()