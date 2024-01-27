---
{"dg-publish":true,"permalink":"/reinforcement-learning/9-imitation-learning/","dgPassFrontmatter":true,"created":"2024-01-10T10:29:37.301+08:00"}
---

代码 [15\_模仿学习.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/15_%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0.ipynb)
# 1. 行为克隆
>behavior clone

用一个专家轨迹作为labels，训练方法和深度学习一样，专家可以是人，也可以是训练好的模型。行为克隆效果一般不会很好，因为每次环境反馈的不会完全一样，只有在专家碰到过的那个行为分布上，才会有比较好的效果，一旦碰到专家没碰到过的情况，智能体就只能随机选择动作，可能会有比较大的错误，这种错误会一直积累，所以泛化性很差，如果收集了大量专家数据并且进行训练，效果可能回比较好。

# 2. 生成对抗模仿学习
>generative adversarial imitation learning，GAIL

前面还是要写 [[Reinforcement Learning/5_PPO\|PPO]] 算法并且定义 agent，要生成专家数据，存储专家状态和动作表，agent 还是要自己和环境交互的，只是在交互中学习，使得 agent 的动作分布逐渐接近专家的动作分布。例如我们需要定义一个 [[Reinforcement Learning/5_PPO\|PPO]] 算法，定义一个 GAIL 算法，里面有判别器。

先收集agent和专家的状态、动作序列，再训练判别器，向判别器分别输入两者的状态、动作序列，判别器给出一个对数化概率`log_prob`，越靠近1代表越像agent，越靠近0则代表越像专家，也就是说agent轨迹的label就是1，专家轨迹的label就是0，因此把两者的交叉熵损失加起来就是判别器的总损失，优化器对这个总损失进行梯度下降训练判别器，需要注意的是，判别器是不需要预训练的，与RL同步训练即可。

更新演员网络，需要把reward修改掉，改成`-log_prob`，也就是agent越不像专家，$prob$就越接近1，奖励就越低，也就是这个概率越接近0越好。而PPO里面要梯度上升，就是要往提升奖励的方向调整，提升`-log_prob`，也就是要$prob$靠近0。