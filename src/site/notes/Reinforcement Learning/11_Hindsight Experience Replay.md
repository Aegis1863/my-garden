---
{"dg-publish":true,"permalink":"/reinforcement-learning/11-hindsight-experience-replay/","dgPassFrontmatter":true,"created":"2024-01-10T10:29:37.197+08:00"}
---

代码 [18\_目标导向强化学习.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/18_%E7%9B%AE%E6%A0%87%E5%AF%BC%E5%90%91%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.ipynb)
# 1. 事后经验回放
#机器学习/强化学习/异策略 

该方法简称 HER，也可以叫事后诸葛亮算法，即虽然有些动作没有获得奖励，但是在学习经验时，仍然会取消一些行为的惩罚甚至给奖励。该方法只能用于异策略算法，如 [[Reinforcement Learning/1_DQN\|DQN]]、[[Reinforcement Learning/6_DDPG\|DDPG]]、[[Reinforcement Learning/7_TD3\|TD3]] 等。

在采样方面，和前面的模型预测控制一样，一次采一整个轨迹，首先在轨迹上随机抽一个状态 $s_1$，再确定一个状态 $s_2$，作为一个新的目标，$s_2$ 的选取有多种方法，这里只说一个：在该轨迹上从 $s_1$ 至最终状态里随机抽一个 $s_2$ ，然后计算 $s_1$ 的下一个状态 $s_1^{'}$ 与 $s_2$ 的距离，设定一个阈值，如果比这个阈值小，就认为虽然没有完成任务，但是已经有所进步，故把 $s_1$ 的 $r_1$ 从惩罚变成不惩罚，或者改成奖励。

所以这个方法只是在经验池和采样方面做了修改，但是效果是不错的。原始情况下，智能体做了大量交互但是可能毫无奖励，因为奖励通常是很稀疏的，所以智能体长期学不到东西，通过 HER 算法把一些一无所获的探索也给出奖励，这样相当于鼓励了探索。

可以发现强化学习的很多改进算法都是在改进探索机制。比如 [[Reinforcement Learning/8_Soft Actor Critic (SAC)\|SAC]] 引入动作熵的概念强调探索和利用的平衡，

