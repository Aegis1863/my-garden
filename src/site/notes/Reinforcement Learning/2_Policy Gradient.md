---
{"dg-publish":true,"permalink":"/reinforcement-learning/2-policy-gradient/","dgPassFrontmatter":true,"created":"2023-08-07T17:24:54.354+08:00"}
---

代码 [09\_梯度策略.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/09_%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6.ipynb)
# 1. 策略网络
#机器学习/强化学习/同策略 #机器学习/强化学习/策略网络 #机器学习/强化学习/离散动作 
#机器学习/强化学习/连续动作

梯度策略也要构建一个神经网络，但是该网络输出的是动作而不是价值，输入的还是state，输出均值和方差去建立一个正态分布，然后抽取动作，如[[Reinforcement Learning/5_PPO\|5_PPO]]算法，或者用softmax输出各个动作的概率，建立一个分类分布，抽取动作例如REINFORCE算法；或者输出一个值用tanh映射到-1~1的区间，再缩放到动作空间，例如[[Reinforcement Learning/6_DDPG\|6_DDPG]]和[[Reinforcement Learning/7_TD3\|7_TD3]]算法。在[[Reinforcement Learning/8_Soft Actor Critic (SAC)\|SAC算法]]中，策略网络输出均值和方差建立一个分布，抽取动作，最后把动作映射到tanh上，经放缩后输出，同时输出一个对数动作概率。

策略网络和Q网络的区别，参考[[Reinforcement Learning/3_Similar Concepts\|3_Similar Concepts]]。

# 2. 目标函数推导

策略梯度的损失，不是与正确的动作来对比得到交叉熵，因为我们事先不知道正确的动作，所以没有参考，因此就只能人为确定一种损失，例如用网络给出的各个动作的概率，乘以当前得到的奖励（执行动作得到的当前奖励），如果奖励比较低，那么这个损失就比较小，我们需要`梯度上升`，所以要加个负号，那么奖励越小，损失就越大，反之同理，符合我们调整参数的目标。

所以目标函数就是奖励，即对于该状态动作序列，动作概率乘以动作奖励

$$
\bar{R}_{\theta}=\sum_{\tau} R(\tau) p_{\theta}(\tau)
$$

梯度策略的更新就是直接对奖励梯度上升
{ #051b94}


$$ \begin{aligned} 
\nabla \bar{R}_{\theta}&=\sum_{\tau} R(\tau) \nabla p_{\theta}(\tau)\\
&=\sum_{\tau} R(\tau) p_{\theta}(\tau) \frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)} \\&= \sum_{\tau} R(\tau) p_{\theta}(\tau) \nabla \log p_{\theta}(\tau) \\
&=\mathbb{E}_{\tau \sim p_{\theta}(\tau)}\left[R(\tau) \nabla \log p_{\theta}(\tau)\right]\\&\approx \frac{1}{N} \sum_{n=1}^{N} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(\tau^{n}\right) \\ 
&=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_{n}} R\left(\tau^{n}\right) \nabla \log p_{\theta}\left(a_{t}^{n} \mid s_{t}^{n}\right) 
\end{aligned}$$

但是 $R$ 需要改为优势函数，因为加分或者扣分可能是由前面某一动作造成的，而不是后来的动作造成的，每个状态每个动作都有不同的结果，所以用一整场游戏的分数对应给某个动作的概率是不合理的，所以每个动作的概率只需要乘以往后的折算的奖励，越远的动作造成的奖励，对当前动作的奖励影响也比较小，所以改写成：
$$\nabla\bar{R}_{\theta}\approx\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_{n}}\left(\sum_{t^{T}=t}^{T_n}\gamma^{t^{\prime}-t}r_{t^{\prime}}^{n}-b\right)\nabla\log p_{\theta}\left(a_{t}^{n}\mid s_{t}^{n}\right)$$
b是基线，一般REINFORCE算法里面没有算b，它可以是奖励的均值，不写也可以，这里对奖励折算，就需要从后往前循环迭代计算，并且每往前推一次就要记录一次梯度。
$$\sum_{t^{T}=t}^{T_n}\gamma^{t^{\prime}-t}r_{t^{\prime}}^{n}-b$$
这部分这里实际上就是优势函数，当然也可以不减b。折算的那部分，可以理解为奖励应该是多少，其实也可以用Q网络时序差分的目标$r+Q(s',a')$来代替，这样就不用折算了，还可以减去一个b，b也可以用Q网络估计出来，这就是`评论员`，评论员估计的是当前的价值是多少，所以优势函数的目的就是计算实得和应得的差异。在后面的演员评论员框架中，演员通常是策略网络，评论员通常是Q网络。
