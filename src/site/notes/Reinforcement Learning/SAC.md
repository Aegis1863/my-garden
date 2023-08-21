---
{"dg-publish":true,"permalink":"/reinforcement-learning/sac/","dgPassFrontmatter":true}
---


# SAC算法
#机器学习/强化学习/异策略 #机器学习/强化学习/连续动作 

SAC模型的目标就是希望策略的回报和熵都尽可能大，熵前面有一个参数调整熵的重要性, 也称为`温度`​。

SAC采用软更新，主要是对两个critic进行软更新，在更新中取给出Q值比较小的那个，类似[[Reinforcement Learning/TD3\|TD3]]的技巧一，对于演员则采用熵约束的方法。
#机器学习/强化学习/软更新 #机器学习/强化学习/熵约束

`动作熵的概念`：对网络输入状态, 网络给出动作和该动作的对数概率, 而把动作的对数概率取负就是熵, 熵越大, 代表该动作的不确定性越大, 也就是发生概率较小, 反过来, 熵越小, 说明确定性越大。

熵约束即既要求回报最大，也要求熵比较大，这就意味着要平衡回报和探索，目标如下
$$
\pi^* = \underset{\pi}{\operatorname{argmax}}\mathbb{E}_{\pi}\left[\underset{t}{\sum}r(s_t,a_t)+\alpha\mathcal{H}(\pi(\cdot|s_t)) \right]
$$
也可以写成总的训练目标
$$
J(\pi)=\sum^{T}_{t=0}\mathbb{E}_{(s_t,a_t)\sim\rho_\pi}[r(s_t,a_t)+\alpha\mathcal{H}(\pi(\cdot|s_t))]
$$

其中熵就是对数化的动作概率加负号, 在代码里面表现为变量`entropy`

$$
\mathcal{H}(\pi(\cdot|s_t)) = -\log(\pi(a|s))
$$

并且V函数也加入熵, 如果不考虑负号应该是加上$\alpha\mathcal{H}(\pi(\cdot|s_t))$, 改写为$V^\pi_{soft}(s)$
$$
V^\pi_{soft}(s^\prime)=\mathbb{E}_{(s^\prime,a^\prime)\sim\rho_\pi}[Q_{soft}(s^\prime,a^\prime)-\alpha\log(\pi(a^\prime|s^\prime))]
$$

因此Q函数的贝尔曼方程跟以前类似 
$$
\begin{align}
Q^\pi_{soft}(s,a) 
& = r(s,a)+\gamma\mathbb{E}_{(s^\prime,a^\prime)\sim\rho_\pi}[Q_{soft}(s^\prime,a^\prime)-\alpha\log(\pi(a^\prime|s^\prime))]\\

& = r(s,a)+\gamma\mathbb{E}_{s^\prime\sim\rho}[V^\pi_{soft}(s^\prime)]
\end{align}
$$

## Q网络的目标函数

对应代码`SAC`类中的`calc_target`方法，也就是在[[Reinforcement Learning/DQN\|Q网络]]的基础上改为软Q网络，公式为

$$
\begin{align}
J_Q(\theta) &= \mathbb{E}_{(s_t,a_t,s_{t+1})\sim \mathcal{D}}\left[\frac{1}{2}(Q_\theta(s_t,a_t)-(r(s_t,a_t)+\gamma V_{\bar{\theta}}(s_{t+1})))^2\right] \\
&=\mathbb{E}_{(s_t,a_t,s_{t+1})\sim \mathcal{D}}\left[\frac{1}{2}(Q_\theta(s_t,a_t)-(r(s_t,a_t)+\gamma (Q_{\bar \theta}(s_{t+1},a_{t+1})-
\alpha\log(\pi(a_{t+1}|s_{t+1})))^2\right]
\end{align}
$$

其中$Q_{\bar \theta}$是两个评论员网络给出的价值评价的较小的那个，这里对$\theta$求梯度是很简单的。

## V网络的目标函数

即策略网络Policy，公式为

$$
\begin{align}
J_\pi(\phi)
&= D_{KL}\left(\pi_k(\cdot|s_t)||\frac{exp(\frac{1}{\alpha}Q^\pi_{soft}(s_t,\cdot))}{Z_{soft}^\pi(s_t)}\right)\\
&= \mathbb{E}_{s_t\sim\mathcal{D},a\sim\pi_\phi}\left[\log\pi_\phi(a_t|s_t)-\frac{1}{\alpha}Q_{\bar \theta}(s_t,a_t)+\log Z(s_t)\right]
\end{align}
$$
其中KL散度公式：
$$
\begin{align}
D_{KL}(P||Q) &= \int_x p(x)\operatorname{log}\frac{P(x)}{Q(x)}\operatorname{d}x \\
&= \mathbb{E}\left[\operatorname{log}\frac{P(x)}{Q(x)} \right]

\end{align}
$$

由于求梯度时也不需要对$Z$求梯度，而且本来也不知道$Z$，所以代码里面去掉了， [原论文](https://arxiv.org/pdf/1801.01290.pdf)思想也就是这样，位于第四页右侧上面，代码里面还全部同时乘以了$\alpha$，这里就不乘了，直接写为

$$
J_\pi(\phi) = \mathbb{E}_{s_t\sim\mathcal{D},a\sim\pi_\phi}\left[\log\pi_\phi(a_t|s_t)-\frac{1}{\alpha}Q_{\bar \theta}(s_t,a_t)\right]
$$

这个目标函数是KL散度，表示两个分布之间的差异程度，减小这个目标函数就是试图使策略函数$\pi_k$的分布看起来更像是由函数Z标准化的Q函数的指数分布；其中$a_t$是重参数化来的，目的是让梯度正常更新，$\epsilon$是从高斯分布采样的噪音。对策略目标函数的参数$\phi$求导时，与$Z$无关，但是与Q有关，因为Q中的$a_t$是从策略中重参数化取得的，如下面公式

$$
a_t=f_\phi(\epsilon_t;s_t)=f^\mu_\phi(s_t)+\epsilon_t\cdot f^\mu_\phi(s_t)
$$

## 动作被压缩后概率密度的变化

在求熵的时候，动作$u$原本是从正态分布采样的，但是经过$a=\operatorname{tanh}(u)$压缩处理，不再是正态分布，需要重新计算概率，需要注意的是，这里采样到的单个动作的概率就是概率密度，而不是概率分布，这里的$\pi$和$\mu$都是概率密度函数，推导如下

$$
\begin{align}
a &= \operatorname{tanh}(u) \\
\pi(a|s)\operatorname{d}a&=\mu(u|s)\operatorname{d}u\\
\pi(a|s) 
&= \mu(u|s)(\frac{\operatorname{d}a}{\operatorname{d}u})^{-1}\\
其中:\frac{\operatorname{d}a}{\operatorname{d}u} &= 1-\operatorname{tanh}^2u \\
\pi(a|s) &= \mu(u|s)(1-\operatorname{tanh}^2u)^{-1} \\
两边加对数:\operatorname{log}\pi(a|s) 
&= \operatorname{log}\mu(u|s)-\sum^D_{i=1}\operatorname{log}(1-\operatorname{tanh}^2(u_i)) \\
&= \operatorname{log}\mu(u|s)-\sum^D_{i=1}\operatorname{log}(1-a_i^2)
\end{align}
$$

## α的目标函数

约束优化问题中没有$\alpha$，它是作为拉格朗日乘子出现的。

存在以下约束优化, 其中$\mathcal{H}_0$是可以调整的参数。

$$
\underset{\pi}{\operatorname{max}} \mathbb{E}\left[\underset{t}{\sum}r(s_t, a_t) \right] ~~~~ \operatorname{s.t.} ~~~~ \mathbb{E}_{(s_t,a_t)\sim\rho_\pi} \left[-\operatorname{log}(\pi(a_t|s_t)) \right] \geq \mathcal{H}_0
$$

参考[第十三章 SAC 算法](https://johnjim0816.com/joyrl-book/#/ch13/main?id=%e8%87%aa%e5%8a%a8%e8%b0%83%e8%8a%82%e6%b8%a9%e5%ba%a6%e5%9b%a0%e5%ad%90)，先不考虑$\sum r(s_t,a_t)$，从最后一项往前推，也就是$t=T$时开始往前推即可发现规律，如下所示（来源[Joy RL](https://johnjim0816.com/joyrl-book/#/ch13/main?id=%e8%87%aa%e5%8a%a8%e8%b0%83%e8%8a%82%e6%b8%a9%e5%ba%a6%e5%9b%a0%e5%ad%90)）

$$\underbrace{\max _{\pi_0}(\mathbb{E}\left[r\left(s_0, a_0\right)\right]+\underbrace{\max _{\pi_1}(\mathbb{E}[\ldots]+\underbrace{\max _{\pi_T} \mathbb{E}\left[r\left(s_T, a_T\right)\right]}_{ \text { 第一次最大（子问题一） }})}_{\text { 倒数第二次最大 }})}_{\text { 倒数第一次最大 }}$$


所以第一次优化是

$$
\begin{align}
&\underset{\pi_{T}}{\operatorname{max}} \mathbb{E}\left[r(s_T, a_T) \right] \\
&\operatorname{s.t.} ~~~~ \mathbb{E}_{(s_T,a_T)\sim\rho_{\pi _T}} \left[-\operatorname{log}(\pi(a_T|s_T)) \right] \geq \mathcal{H}_0
\end{align}
$$

拉格朗日函数为
$$
\begin{align}
L(\pi_T,\alpha_T)
&=\mathbb{E}\left[r(s_T, a_T) + \alpha_T(-\operatorname{log}(\pi(a_T|s_T)))\right]-\alpha_T\mathcal{H_0}\\
其中~~~~~~~
\mathcal{H}(\pi_T) &= -\operatorname{log}(\pi(a_T|s_T))\\
所以~~L(\pi_T,\alpha_T)&= \mathbb{E}\left[r(s_T, a_T) +\alpha_T\mathcal{H}(\pi_T)\right]-\alpha_T\mathcal{H_0}\\
\end{align}
$$
转化为对偶问题

$$
\begin{align}
\underset{\pi_T}{\operatorname{max}}\mathbb{E}(r(s_T, a_T)) 
&= \underset{\alpha_{T}\geq0}{\operatorname{min}}\underset{\pi_T}{\operatorname{max}}\mathbb{E}\left[r(s_T, a_T) +\alpha_T\mathcal{H}(\pi_T)\right]-\alpha_T\mathcal{H_0}\\

先把\alpha_T固定住不&动得到最佳策略\\
\pi_T^* &= \underset{\pi_T}{\operatorname{argmax}}\mathbb{E}(r(s_T, a_T)+\alpha_T\mathcal{H}(\pi_T)-\alpha_T\mathcal{H_0})\\

再确定\alpha_T\\

\alpha_T^* &= \underset{\alpha_T}{\operatorname{argmax}}\mathbb{E}(r(s_T, a_T)+\alpha_T\mathcal{H}(\pi^*_T)-\alpha_T\mathcal{H_0})\\
\end{align}
$$

这里发现$\pi_T^*$和$\alpha_T^*$的目标式是一样的，如果不清楚目标式是怎么来的，参考下面解$\underset{\alpha\geq0}{\operatorname{min}}\underset{\pi_T}{\operatorname{max}}$问题的步骤：

1. 先固定外面的变量，看内层，就是说带外层参数的项都不动，这里指$\alpha_T$，带内层参数$\pi_T$这一项的也不动，但是可以去掉等于0的项或者常数——因为是$\operatorname{argmax}$求参数，要确定这个参数，和式子里面常数项的大小无关，和等于0的项也无关，只和带参数的项有关——这样就得到了关于内层参数的目标式，就是说确定了该参数的取值；
2. 第二步是把第一步已经确定的这个参数固定住，当成常数，看到外层，式子里面保留含有外层参数的项，而常数项和等于0的项都去掉，可以看到$\mathcal{H}(\pi^*_T)$和$\mathcal{H}_0$都是常数，但是它们都带了系数$\alpha_T$，也就是带了参数，所以不能去掉。

所以经过上面两步，就得到了目标式，但是还没结束，因为这只是第一项的，但是往后推也一样，第二个优化问题就是

$$
\begin{align}
&\underset{\pi_{T-1}}{\operatorname{max}} \mathbb{E}\left[r(s_{T-1}, a_{T-1})+r(s_{T}, a_{T}) \right] \\ 
&\operatorname{s.t.} ~~ \mathbb{E}_{(s_{T-1},a_{T-1})\sim\rho_{\pi_{T-1}}} \left[-\operatorname{log}(\pi(a_{T-1}|s_{T-1})) \right] \geq \mathcal{H}_0
\end{align}
$$
--------------------------插入内容----------------------------

此时，有$Q_{soft}$函数的迭代公式如下，式子里面就不写$soft$了
$$Q_{T-1}(s_{T-1}, a_{T-1}) = r(s_{T-1}, a_{T-1})+\gamma\mathbb{E}[r(s_T,a_T)+\alpha_T\mathcal{H}(\pi_T)]$$
上式是标准的$Q_{soft}$值目标式，但是在这里的操作中，$\gamma$被去掉了，[提出自动调节温度因子的SAC原论文](https://arxiv.org/pdf/1812.05905.pdf)就没有写。

**个人想法**：实际上不写$\gamma$影响不大，因为就算保留，最后也只是影响$\mathcal{H}_0$，而目标熵是人为固定的，没必要再乘以一个折扣了。所以去掉$\gamma$，可以推出

$$Q_{T-1}(s_{T-1}, a_{T-1})-\alpha_T\mathcal{H}(\pi_T) = r(s_{T-1}, a_{T-1})+\mathbb{E}[r(s_T,a_T)]$$

**Question**：等式右边就是上面第二个优化问题的目标，相当于做了个代换，为什么要有这个代换呢？如果不代换，就把原式的reward项当常数项去掉，结果也是一样的。

**个人想法**：因为这样可以把第一个优化中已经求出来的 $\pi_T$ 带入$\mathcal{H}(\pi_T)$ ，并且 $Q_{T-1}(s_{T-1}, a_{T-1})$ 根据第一次优化得出的结果，可以求出来，所以这两个都是已知的常数项了。在第一个优化问题中，[原论文公式（14）处](https://arxiv.org/pdf/1812.05905.pdf)也直接去掉了reward项，但是在[Joy RL的推导](https://johnjim0816.com/joyrl-book/#/ch13/main?id=%e8%87%aa%e5%8a%a8%e8%b0%83%e8%8a%82%e6%b8%a9%e5%ba%a6%e5%9b%a0%e5%ad%90)中，reward保留了，本文参考沿用该写法，本文认为这样更严谨——因为reward并不是人为确定的，所以不能视为常数，而Q函数和熵都是可以通过已知数据人为算出来的。

--------------------------插入结束----------------------------

因此可以转化为下面这个$\operatorname{max}$式子，所以在下面式子第二个等号转化为对偶函数时，加上拉格朗日乘子，注意：带星号的都视为常数，有

$$
\begin{align}
&\operatorname{max}(Q^*_{T-1}(s_{T-1},a_{T-1})-\alpha^*_T\mathcal{H}(\pi_T^*))\\
&= \underset{\alpha_{T-1}\geq0}{\operatorname{min}}\underset{\pi_{T-1}}{\operatorname{max}}
(Q^*_{T-1}(s_{T-1},a_{T-1})-\alpha^*_T\mathcal{H}(\pi_T^*))+\alpha_{T-1}(\mathcal{H}(\pi_{T-1})-\mathcal{H}_0)\\



\end{align}
$$
先确定$\pi_{T-1}^*$，再去掉无关的常数，就可以给出$\alpha_{T-1}$的目标式

$$
\alpha_{T-1}=\operatorname{argmax}
\mathbb{E}_{a_t\sim\pi_t}\left[\alpha(\mathcal{H}(\pi_{T-1}^*)-\mathcal{H}_0)\right]
$$

发现和第一个优化问题的解的形式是一样的，于是给出其目标函数为
$$

\begin{align}

J(\alpha)

&= \mathbb{E}_{a_t\sim\pi_t}\left[-\alpha \log \pi_t(a_t|s_t)-\alpha\mathcal{H}_0\right] \\

&= \mathbb{E}_{a_t\sim\pi_t}\left[\alpha(-\log \pi_t(a_t|s_t)-\mathcal{H}_0)\right]

\end{align}

$$
其中$-\log \pi_t(a_t|\pi_t)$也就是熵`entropy`，对$\alpha$求导，就等于小括号里面的动作熵减去目标熵，所以这就是要使得动作熵靠近目标熵，使得$J(\alpha)$变小。

参考下面策略目标函数，动作熵太大，大于目标熵了，梯度就是正的，$\alpha$目标函数在梯度下降时就会减小$\alpha$，就是指策略给的动作的概率比较低，拿不准应该怎么办，需要减小$\alpha$使得模型专注于获得更高的回报，也就是尽可能提升动作确定性；而动作熵太小，就意味着策略对某一个动作过于肯定，容易造成过拟合，就调整$\alpha$变大，$\alpha$变大就倾向于要求动作熵变大，等于希望动作概率降低，让模型重视探索。

$$
\pi^* = \underset{\pi}{\operatorname{argmax}}\mathbb{E}_{\pi}\left[\underset{t}{\sum}r(s_t,a_t)+\alpha\mathcal{H}(\pi(\cdot|s_t)) \right]
$$
## 目标熵的设定 

 [原论文](https://arxiv.org/pdf/1801.01290.pdf)认为连续动作的目标熵等于动作空间的负值，torchrl中默认为`torch.prod(n_actions)`，另一篇[论文](https://arxiv.org/pdf/1910.07207.pdf)给出离散SAC，目标熵设置为$0.98(-\operatorname{log}(\frac{1}{|A|}))$，依据未说明，可能和交叉熵类似，但是总的来说不建议将SAC应用于离散动作环境。