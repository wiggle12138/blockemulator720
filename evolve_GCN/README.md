# 说明
- 时序嵌入从temporal_embeddings.pkl中读取。
- 邻接矩阵从step1_adjacency_raw.pt中读取。
- 在config.py中调整参数，运行train.py即可开始训练。

# 现有的问题
- 分片结果会存在空片，预测分片数和实际分片过程配合得不好。
- 因为模拟了一些时间步，还有其他的随机部分，虽然设置了随机种子，但结果仍然难以复现。
- 是否符合全步骤的设计还需要进一步考察。
- ....