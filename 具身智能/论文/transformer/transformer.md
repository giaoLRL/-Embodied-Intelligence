# Transformer
## Transformer 架构完整执行流程
Transformer 是 2017 年《Attention is All You Need》提出的纯注意力机制的 seq2seq 序列建模架构，核心为 ** 编码器 - 解码器（Encoder-Decoder）** 的对称结构，原始论文中编码器、解码器均为 6 层堆叠，彻底替代了 RNN/CNN 的时序建模方案，实现了长距离依赖捕捉与训练并行化。
完整执行流程分为5 大核心阶段，全程保持核心维度d_model=512（原始论文标准）不变，以下按前向传播的时序顺序，逐层拆解每一步的输入、操作、输出与核心逻辑。
