import torch
from torch import nn
from torchnlp.word_to_vector import GloVe

word_to_ix = {"hello":0, "world": 1}

lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)

"""
创建一个嵌入层，词汇表大小为2，嵌入维度为5。即2个单词，5维向量
"""
embeds = nn.Embedding(2, 5)
hello_embed = embeds(lookup_tensor)
# tensor([[ 0.1816, -0.1760, -0.9428, -0.0661,  0.9271]],
#        grad_fn=<EmbeddingBackward0>)
print(hello_embed)

print('------------------')
"""
GloVe的核心思想基于这样一个假设：单词的意义可以通过它们共同出现的频率来揭示，即共现频率高的单词在语义上更加接近。
GloVe通过构建一个全局共现矩阵来表示单词之间的共现关系，然后利用矩阵分解技术来学习单词的向量表示。
具体来说，GloVe模型首先统计语料库中单词对的共现频率，并构建一个共现矩阵X，其中X_ij表示单词i和单词j在一定窗口大小内共同出现的次数。
然后，GloVe模型试图学习单词向量，使得它们的点积等于它们的共现概率的对数值。
"""
vectors = GloVe()
print(vectors['hello'].shape)