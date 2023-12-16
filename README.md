# 🪄Meta-Path-Based-Random-Walk

**An Attentional Neural Networkfor Personalized Next Location Recommendation** 这篇论文给出了基于元路径的随机游走算法描述，但是并没有开源相关代码，所以笔者在这里给出具体的实现方法，用于论文的复现。但是目前运算性能效率欠佳，仍需改进。

## Method
首先描述一下随机游走（random walk）模型，给定一个含有n个节点都有向图，在有向图上定义随机游走，也就是一阶马尔可夫链，节点可用来表示状态，有向边表示状态之间的转移。

这里有一个假设，从一个节点通过有向边相连的所有节点的转移概率相等，当节点的type相同时，可以将转移关系描述为n阶的转移矩阵$M$。
<center>
<img src="https://latex.codecogs.com/svg.image?M&space;=&space;[m_{ij}]_{n\times&space;n}" title="M = [m_{ij}]_{n\times n}" />
</center>

它描述的是列下标对应的节点转移至行下标对应的节点的概率，换句话说，$m_{ij}$是j节点指向i节点的概率。那么它的性质也有了。第一个共识很显然，第二个就是如果存在由j指向i的有向边，那么该位置的值位j节点的出度分之一，理由就是他们是等概率的。

<center>
<img src="https://latex.codecogs.com/svg.image?\sum_{i&space;=&space;1}^{n}m_{ij}&space;=&space;1" title="\sum_{i = 1}^{n}m_{ij} = 1" />
</center>
<center>
<img src="https://latex.codecogs.com/svg.image?if~~j->i~:~m_{ij}&space;=&space;\frac{1}{d^{&plus;}_{j}}" title="if~~j->i~:~m_{ij} = \frac{1}{d^{+}_{j}}" />
</center>

这个矩阵$M$也叫做随机矩阵（stochastic matrix）。

可以理解，$M$是用来进行表征节点的转移偏好的，但游走过程不仅由转移偏好决定，同时也受转移的起点决定。所以提出一个n维向量$V_t$来表征t时刻转移前，本次转移过程的初始节点的概率分布。
<center>
<img src="https://latex.codecogs.com/svg.image?V_t&space;\epsilon&space;\mathbb{R}^{n&space;\times&space;1}" title="V_t \epsilon \mathbb{R}^{n \times 1}" />
</center>

那么：
<center>
<img src="https://latex.codecogs.com/svg.image?V_{t&plus;1}&space;=&space;M&space;V_{t}" title="V_{t+1} = M V_{t}" />
</center>

这里得到的t+1时间步的V就是时间步t这次转移活动在$V_t$初始分布前提下的转移结果分布，因此，可以以这种方式进行迭代，从而进行多次游走。

那么基于**元路径**的随机游走，大体与之相同，但是也有区别。既然要引入meta path的概念，那么图中节点的种类就不是唯一的，在ARNN要解决的任务中，图中存在三类节点，L:地点，U：访问者，V：地点种类，他们构成图。而基于“LL”、“LVL”、“LUL”这三类元路径的路径都要分别以他们为路径元素type的最小重复单元。

也就是每次转移的随机矩阵是需要不同的，“LL”就与上面讲的一样，而对“LVL”而言，需要两个随机矩阵，分别是(num_v, n)与(n, num_v)，前者与地点分布向量相乘（单个path起点loc为1，其余均为0），从而得到type为v的节点概率分布向量，后者再与地点种类分布向量相乘，又得到地点分布向量，然后继续如此迭代。

“LUL”也是如此。

需要注意的是，这里我虽然将三类节点统一编码，并用三元组构成图谱，但并没有将所有类型的节点放在同一个tensor里，而是meta path在当前需要什么类型，我就单独把起点与终点的类型的节点构成tensor来进行计算，拓扑上讲就是讲图拆分，但是概率依赖关系不受影响。因为我认为所有实体的个数作为tensor的大小用来计算，效率会很低，不如拆分成多组tensor，直观且高效。

思路就是这样，代码实现方面，一开始按照<a href="https://github.com/xinbowu2">@Xinbo Wu</a>复现Personalised Page Rank的方法编写，使用dict来进行矩阵运算，结果显然是差强人意的，面对foursquare的数据运算效率就已经无法接受了，因此使用tensor放到GPU上进行运算，结果明显快了很多，效率勉强让人接受，其实也许可以通过解决矩阵稀疏的问题再加快游走效率，笔者目前能力有限，没找到能更加提高效率的办法，希望大家给予思路。

参考《统计学习方法》李航