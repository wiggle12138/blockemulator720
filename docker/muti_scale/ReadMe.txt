======================================================================
Blockchain Multi-Scale Contrastive Learning Model
======================================================================
Strategy: Fixed learning rate 0.02
Temperature: 0.12 (enhanced contrast)
Target: Loss below 0.25
======================================================================
Using device: cpu
Classic feature data shape: (200, 129)
Number of classic feature columns: 128
Graph data loaded, number of nodes: 200
Created virtual timestamps, treating graph as static snapshot
Dataset initialization complete:
- Number of nodes: 200
- Feature dimension: 128
- Number of edges: 4485
- Number of node types: 5
Starting training with fixed learning rate...
Epoch 10/300 | Loss: 0.5642 | Best: 0.5642 | LR: 0.020000 (fixed) | Patience: 0/40
Epoch 20/300 | Loss: 0.5009 | Best: 0.4991 | LR: 0.020000 (fixed) | Patience: 1/40
Epoch 30/300 | Loss: 0.4812 | Best: 0.4747 | LR: 0.020000 (fixed) | Patience: 5/40
Epoch 40/300 | Loss: 0.4649 | Best: 0.4520 | LR: 0.020000 (fixed) | Patience: 2/40
Epoch 50/300 | Loss: 0.4674 | Best: 0.4338 | LR: 0.020000 (fixed) | Patience: 8/40
Epoch 60/300 | Loss: 0.4309 | Best: 0.4100 | LR: 0.020000 (fixed) | Patience: 9/40
Epoch 70/300 | Loss: 0.4124 | Best: 0.3979 | LR: 0.020000 (fixed) | Patience: 6/40
Epoch 80/300 | Loss: 0.4322 | Best: 0.3917 | LR: 0.020000 (fixed) | Patience: 2/40
Epoch 90/300 | Loss: 0.4051 | Best: 0.3848 | LR: 0.020000 (fixed) | Patience: 3/40
Epoch 100/300 | Loss: 0.4003 | Best: 0.3644 | LR: 0.020000 (fixed) | Patience: 1/40
Epoch 110/300 | Loss: 0.3645 | Best: 0.3594 | LR: 0.020000 (fixed) | Patience: 6/40
Epoch 120/300 | Loss: 0.4157 | Best: 0.3506 | LR: 0.020000 (fixed) | Patience: 6/40
Epoch 130/300 | Loss: 0.3942 | Best: 0.3449 | LR: 0.020000 (fixed) | Patience: 9/40
Epoch 140/300 | Loss: 0.3558 | Best: 0.3228 | LR: 0.020000 (fixed) | Patience: 7/40
Epoch 150/300 | Loss: 0.3413 | Best: 0.3228 | LR: 0.020000 (fixed) | Patience: 17/40
Epoch 160/300 | Loss: 0.3482 | Best: 0.3221 | LR: 0.020000 (fixed) | Patience: 4/40
Epoch 170/300 | Loss: 0.3580 | Best: 0.2912 | LR: 0.020000 (fixed) | Patience: 6/40
Epoch 180/300 | Loss: 0.3290 | Best: 0.2912 | LR: 0.020000 (fixed) | Patience: 16/40
Epoch 190/300 | Loss: 0.3256 | Best: 0.2693 | LR: 0.020000 (fixed) | Patience: 1/40
Epoch 200/300 | Loss: 0.2636 | Best: 0.2636 | LR: 0.020000 (fixed) | Patience: 0/40
Epoch 210/300 | Loss: 0.3074 | Best: 0.2636 | LR: 0.020000 (fixed) | Patience: 10/40
Epoch 220/300 | Loss: 0.2994 | Best: 0.2636 | LR: 0.020000 (fixed) | Patience: 20/40
Epoch 230/300 | Loss: 0.3068 | Best: 0.2636 | LR: 0.020000 (fixed) | Patience: 30/40
Epoch 240/300 | Loss: 0.2946 | Best: 0.2548 | LR: 0.020000 (fixed) | Patience: 8/40
Epoch 250/300 | Loss: 0.2846 | Best: 0.2548 | LR: 0.020000 (fixed) | Patience: 18/40
Epoch 260/300 | Loss: 0.2748 | Best: 0.2428 | LR: 0.020000 (fixed) | Patience: 5/40
Epoch 270/300 | Loss: 0.2918 | Best: 0.2428 | LR: 0.020000 (fixed) | Patience: 15/40
Epoch 280/300 | Loss: 0.2767 | Best: 0.2428 | LR: 0.020000 (fixed) | Patience: 25/40
Epoch 290/300 | Loss: 0.2672 | Best: 0.2359 | LR: 0.020000 (fixed) | Patience: 9/40
Epoch 300/300 | Loss: 0.2683 | Best: 0.2278 | LR: 0.020000 (fixed) | Patience: 9/40
Training completed!
Generating final embeddings...
Classic feature data shape: (200, 129)
Number of classic feature columns: 128
Graph data loaded, number of nodes: 200
Created virtual timestamps, treating graph as static snapshot
Dataset initialization complete:
- Number of nodes: 200
- Feature dimension: 128
- Number of edges: 4485
- Number of node types: 5
Embeddings saved to temporal_embeddings.pkl
Training history saved to training_history.pkl
Model saved to mscia_model.pth
Generating visualizations...
Visualizing 200 node embeddings
Embedding visualization saved to visualizations/blockchain_embeddings_t-sne.png
Clustering visualization saved to visualizations/blockchain_embeddings_clusters.png

======================================================================
Training Summary:
Final loss: 0.2683
Best loss: 0.2278
Recent trend: increasing
Target achieved: Yes
Number of node embeddings: 200
Embedding dimension: 64
======================================================================