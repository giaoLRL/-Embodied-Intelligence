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

position 0：

[0,1,0,1]

position 1：

[0.84,0.54,0.01,0.999]

position 2：

[0.91,−0.42,0.02,0.999]

#### 每个位置都有唯一向量
#### 直接加到 token embedding 上
比如，I 的 embedding是：[1.2,0.5,−0.7,0.3]，而position 0：[0,1,0,1]，相加：[1.2,1.5,−0.7,1.3]，将Token向量混入了位置信息。

