---
{"dg-publish":true,"permalink":"/reinforcement-learning/5-ppo/","dgPassFrontmatter":true,"created":"2023-08-07T17:27:23.383+08:00","updated":"2023-10-22T18:56:44.233+08:00"}
---

代码 [12\_PPO.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/12_PPO.ipynb)

#机器学习/强化学习/同策略 #机器学习/强化学习/连续动作 #机器学习/强化学习/离散动作 
# 1. 网络结构

和优势演员评论员机制一样，评论员采用[[Reinforcement Learning/1_DQN\|Q网络]]，演员采用[[Reinforcement Learning/2_Policy Gradient\|梯度网络]]。

不同点在于引入重要性采样和裁剪，重要性采样使得内部可以使用异策略的方法，让 state 不具有序列相关性，而且通过裁剪的方法确保了每次更新幅度不会过大，通过多次学习可以比较精细地调整。

# 2. 重要性采样
{ #86830f}


PPO 限制了更新的大小，称为 `裁剪`，因此对于同一批数据可以多次训练，相当于局部的异策略学习，这样数据利用率会比较高。

但是在内部的每一次训练中，实际上只有第一次是在线学习，因为第一次更新采用的经验就是当前策略的，但是已经更新一次之后，策略就变了，而数据还是之前的，这时就类似离线学习了，为了确保新策略和原来的策略不会有太大偏差，就加了`重要性采样`。在PPO算法内部的更新，严格来说还是离线学习，尽管是进行一整轮游戏之后采取更新。但是由于这种离线学习是建立在在线学习的大框架里面的，所以一般说PPO是在线学习，即同策略。
#机器学习/强化学习/重要性采样 #机器学习/强化学习/异策略 

对于重要性采样，推导如下
$$
\begin{align}
&\displaystyle\int f(x)p(x)\mathrm{d}x=\int f(x)\frac{p(x)}{q(x)}q(x)\mathrm{d}x=\mathbb{E}_{x\sim q}[f(x)\frac{p(x)}{q(x)}] \\

&\mathbb{E}_{x\sim p}[f(x)]=\mathbb{E}_{x\sim q}\left[f(x){\frac{p(x)}{q(x)}}\right]
\end{align}
$$
我们再纳入优势函数A，优势函数的推导参考[[Reinforcement Learning/4_Advantage Actor Critic (A2C)#优势函数推导\|优势函数推导]]，此时可以写出梯度的公式，推导参考[[Reinforcement Learning/2_Policy Gradient\|#目标函数推导]]

$$
\mathbb{E}_{(s_t,a_t)\sim\pi_{\theta^{\prime}}}\left[{\frac{p_{\theta}\left(s_t,a_t\right)}{p_{\theta^{\prime}}\left(s_t,a_t\right)}}A^{\theta}\left(s_t,a_t\right)\nabla\log p_{\theta}\left(a_{t}|s_{t}\right)\right]
$$
那么根据$\nabla f(x)\,=\,f(x)\,\nabla\log f(x)$，有目标函数

$$
J^{\theta^{\prime}}(\theta)=\mathbb{R}_{(s_t,a_ t)\sim \pi_{\theta^{\prime}}}\left[{\frac{p_{\theta}\left(a_t|s_t\right)}{p_{\theta^{\prime}}\left(a_t|s_{t}\right)}}A^{\theta^{\prime}}\!\left(s_t,a_t\right)\right]
$$

# 3. PPO截断
#机器学习/强化学习/裁剪

再加上截断技巧，演员网络目标函数如下（未取均值），应使其最大化，因此代码中还加了负号

![](https://s2.loli.net/2023/07/23/6YZFnoAxfiwX57q.png)

原本是重要性乘以优势函数再乘以演员动作概率的对数，但是与环境交互的演员变成另一个演员了，这个演员在代码中就是原本那个每次训练刚开始时候的初始状态的演员，相当于一个原始副本。

根据A的正负不同，有如下图示，蓝色虚线表示裁剪范围，红色表示最终最小值输出

![|725](https://datawhalechina.github.io/easy-rl/img/ch5/5.3.png)
