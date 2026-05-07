# Transformer
## Transformer 架构完整执行流程
Transformer 是 2017 年《Attention is All You Need》提出的纯注意力机制的 seq2seq 序列建模架构，核心为 **编码器 - 解码器（Encoder-Decoder）** 的对称结构，原始论文中编码器、解码器均为 6 层堆叠，彻底替代了 RNN/CNN 的时序建模方案，实现了长距离依赖捕捉与训练并行化。
完整执行流程分为5 大核心阶段，全程保持核心维度d_model=512（原始论文标准）不变，以下按前向传播的时序顺序，逐层拆解每一步的输入、操作、输出与核心逻辑。

### 阶段 1：输入序列预处理（Token 嵌入 + 位置编码）
Transformer 的输入是离散的文本序列，首先需要将其转换为模型可处理的稠密向量，并显式注入序列的位置信息（无循环结构，无法天然捕捉时序）。
#### 步骤 1：Token 化与 ID 映射

输入文本通过**分词器**（如 BPE 子词分词）拆分为独立 Token，例如输入I love AI拆分为["I", "love", "AI"]；

将每个 Token 映射为词表中对应的整数 ID，得到形状为[batch_size, src_seq_len]的整数序列，其中：

batch_size：批次样本数，src_seq_len：源序列长度。

Tokenizer 后：
```
[1, 5, 9]
```
假设：
- vocab size = 10000
- embedding dim = 4


#### 步骤 2：Token Embedding（词嵌入）

核心操作：通过**可学习的嵌入矩阵**（形状[vocab_size, d_model]，vocab_size为词表大小），将每个 Token ID 映射为d_model=512维的**稠密向量**；

> 本质就是**查表**

细节补充：嵌入结果会乘以√d_model做缩放，匹配后续位置编码的数值量级；

输出形状：[batch_size, src_seq_len, d_model]。

比如：
```
I     → [1,0,1,0]
love  → [0,1,1,0]
AI    → [1,1,0,1]
```
于是：

输入矩阵：

X∈R^3×4

即：
```
3 个 token
每个 token 4维
```
#### 步骤 3：Positional Encoding（位置编码）

核心目的：显式注入 Token 在序列中的绝对位置与相对位置信息，弥补 Transformer 无循环结构的缺陷；

**原因**：Transformer的Attention本质完全对称，不知道谁在前谁在后，例如"I love you"和"you love I"，如果没有位置编码，Attention只看到三个token向量，并不知道顺序不同。

> RNN天生有顺序，处理：I → love → AI

原始方案：固定正余弦位置编码，公式如下，其中pos为 Token 位置索引，i为向量维度索引：

![PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)](https://latex.codecogs.com/svg.latex?PE_{(pos,2i)}%20=%20\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right))


![PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)](https://latex.codecogs.com/svg.latex?PE_{(pos,2i+1)}%20=%20\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right))

核心操作：位置编码的形状与词嵌入完全一致[batch_size, src_seq_len, d_model]，与词嵌入逐元素相加（非拼接），得到编码器的最终输入；

补充：也可使用可学习位置编码，效果与固定编码相当，主流预训练模型多采用可学习方案。

假设：

d_model = 4

position 0：[0,1,0,1]

position 1：[0.84,0.54,0.01,0.999]

position 2：[0.91,−0.42,0.02,0.999]

#### 每个位置都有唯一向量
#### 直接加到 token embedding 上
比如，I 的 embedding是：[1.2,0.5,−0.7,0.3]，而position 0：[0,1,0,1]，相加：[1.2,1.5,−0.7,1.3]，将Token向量混入了位置信息。

#### 为什么要用sin和cos?
作者希望模型能感知相对位置，不仅是“你在第5个”还包括“你距离我2格”。

sin和cos有个神奇的性质，三角函数具有周期规律，于是不同位置之间会形成可**计算的线性关系**，即模型可以通过线性变化推断token间距离关系。

#### 为什么不同维度频率不同？
公式里：
$$10000^{2i/d_{\text{model}}}$$

作用实时不同维度使用不同波长，低维变化快，高纬变化慢。于是模型就能获得，**短距离位置感知和长距离位置感知**。非常像傅里叶展开。

#### 为什么后来的GPT不用sin和cos？
因为后来发现，可学习位置编码更简单有效。

直接训练
```
position 0 的向量
position 1 的向量
...

```
而不是手工sin和cos。

#### 更先进的位置编码
现代LLM很多已经不用传统absolute position。例如RoPE（Rotary Position Embedding），是Llama/Qwen/GPT-NeoX 常用。

核心思想：不把位置加进去

而是：直接旋转Q/K向量

对上下文效果更好

### 编码器（Encoder）堆叠前向传播
> 原始架构为6 个完全相同的编码器层堆叠，每层输入为上一层输出，全程维度保持d_model不变

#### 编码器整体输入输出
【编码器整体输入】阶段 1 步骤 3 输出的源序列编码向量，形状[batch_size, src_seq_len, d_model]

> ### 形状的作用
> #### 1. 形状 = 操作的 “执行说明书”
> 
> Transformer 的多头注意力，本质上是把数学上的公式翻译成张量运算，而张量的维度直接决定了：
> - 哪一维是并行的头（h）
> - 哪一维是序列长度（src_seq_len）
> - 哪一维是特征维度（d_model/d_k）
> 比如步骤 2 里的形状变化：
> 
> [batch_size, src_seq_len, d_model] → [batch_size, h, src_seq_len, d_k]
> 
> 这个形状变化，直接对应了 “把 d_model 拆成 h 个头，每个头 d_k 维度” 的操作，没有形状，你根本不知道多头是怎么 “并行” 起来的。
>
> #### 2. 形状是理解 “多头并行” 的关键
> Transformer 里的 “多头”，不是真的 8 个独立的网络，而是在张量维度上做了 “拆分”，让所有头的计算能在一次矩阵乘法里完成。
> - 拆分前：[batch, seq_len, d_model]，特征是 “一整块”
> - 拆分后：[batch, h, seq_len, d_k]，h这个维度就代表了 8 个并行的头，后续的注意力计算会自动在这个维度上广播，相当于 8 个头同时计算。
>   
> 如果不看形状，你很难理解 “多头并行” 到底是怎么实现的。
>
> #### 3. 形状决定了矩阵乘法的兼容性
> 缩放点积注意力的核心是 Q @ K.transpose(-2, -1)，这个操作对形状的要求非常严格：
> - Q 的形状是 [batch, h, seq_len_q, d_k]
> - K.transpose 后的形状是 [batch, h, d_k, seq_len_k]
> 
> 只有这样，它们的矩阵乘法才能得到 [batch, h, seq_len_q, seq_len_k] 的注意力分数矩阵。
如果形状不对，比如你把h放在了最后一维，矩阵乘法就会直接报错。
>
> 4. 形状是工程实现和调试的基础
> 写过 Transformer 代码的人都懂：
> 注意力不收敛、输出不对，大概率是形状不匹配
> 多头拆分 / 拼接的时候，维度搞反了（比如把seq_len和h搞混），整个模型就会完全失效
> 反复强调形状，就是帮你提前避开这些 “坑”，理解代码里的view、transpose操作到底在干嘛。

【编码器整体输出】源序列全局语义表征，形状[batch_size, src_seq_len, d_model]

单个编码器层 完整执行步骤（每层重复执行）

子层 1：多头自注意力（Multi-Head Self-Attention, MHSA）

步骤 1：Q/K/V 线性映射

【输入】编码器层原始输入x，形状[batch_size, src_seq_len, d_model]

【核心操作】通过 3 个独立的可学习线性层，分别将输入映射为查询 Q、键 K、值 V 三个矩阵

> QKV三个字母对应三个空间，查询目标空间，标签空间，以及内容空间。大致流程：找一本书，找一本神经网络相关的书，看到书上的内热。

【输出】Q、K、V 矩阵，三者形状均为：[batch_size, src_seq_len, d_model]

步骤 2：多头拆分

【输入】步骤 1 输出的 Q、K、V 矩阵，形状[batch_size, src_seq_len, d_model]

【核心操作】沿最后一维（d_model）将 Q/K/V 均匀拆分为h=8个并行头，每个头维度d_k=64

【输出】拆分后的 Q、K、V，形状均为：[batch_size, h, src_seq_len, d_k]

步骤 3：并行缩放点积注意力计算

【输入】步骤 2 拆分后的 Q、K、V，形状[batch_size, h, src_seq_len, d_k]；Padding Mask 掩码矩阵

【核心操作】每个头独立执行缩放点积注意力计算，公式：

对于注意力机制的**第 $i$ 个头**，独立执行缩放点积注意力：

$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left( \frac{Q_i K_i^T}{\sqrt{d_k}} \right) V_i
$$

### 符号定义
- $Q_i \in \mathbb{R}^{n \times d_k}$：第 $i$ 个头的查询矩阵
- $K_i \in \mathbb{R}^{n \times d_k}$：第 $i$ 个头的键矩阵
- $V_i \in \mathbb{R}^{n \times d_v}$：第 $i$ 个头的值矩阵
- $d_k$：每个头的键维度（$\boldsymbol{d_k = d_{\text{model}} / h}$）
- $h$：注意力头的总数
- $d_{\text{model}}$：模型总维度

1.计算 Q 与 K 的点积，得到 Token 间相似度矩阵

2.除以√d_k做数值缩放，避免 softmax 梯度消失

3.通过 Padding Mask 将无效补零位置的注意力权重置为-∞，softmax 后概率趋近于 0

4.注意力权重与 V 矩阵相乘，得到单头注意力输出
【输出】8 个并行头的注意力结果，形状均为：[batch_size, h, src_seq_len, d_k]
