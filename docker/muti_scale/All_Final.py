import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
# from MS_CIA.MS_CIA import MSCIA


class NodeSamplingHead(nn.Module):
    def __init__(self, gnn, mlp, k_ratio):
        super().__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.k_ratio = k_ratio

    def forward(self, A, X):
        # Step 1: Get node embeddings
        H_v, h_G = self.gnn(A, X)  # [n, d], [1, d]

        # Step 2: Gumbel-TopK node sampling
        n = A.size(0)
        num_keep = max(1, int(self.k_ratio * n))

        node_logits = self.mlp(H_v)  # Node importance scores
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(node_logits) + 1e-8) + 1e-8)
        perturbed_logits = node_logits + gumbel_noise

        _, topk_indices = torch.topk(perturbed_logits.squeeze(),
                                     k=num_keep, sorted=False)

        # Step 3: Build subgraph (keep only sampled nodes and their edges)
        node_mask = torch.zeros(n, dtype=torch.bool, device=A.device)
        node_mask[topk_indices] = True

        # New adjacency matrix (only edges between sampled nodes)
        A_aug = A.clone()
        A_aug[~node_mask, :] = 0
        A_aug[:, ~node_mask] = 0

        return A_aug


class EdgeSamplingHead(nn.Module):
    def __init__(self, gnn, mlp, k_ratio):
        super().__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.k_ratio = k_ratio

    def forward(self, A, X):
        # Step 1: Get node embeddings and graph summary
        H_v, h_G = self.gnn(A, X)  # H_v: [n, d], h_G: [1, d]

        # Step 2: Build edge feature candidates
        n = A.size(0)
        device = A.device
        C_e = []
        edge_positions = []  # Record edge positions
        
        for i in range(n):
            for j in range(n):
                if A[i, j] != 0:  # Only process existing edges
                    edge_feat = torch.cat([
                        H_v[i] + H_v[j],  # Node embedding sum
                        h_G.squeeze(0),  # Graph summary
                        torch.tensor([1.0], device=device)  # Edge existence indicator
                    ])
                    C_e.append(edge_feat)
                    edge_positions.append((i, j))
        
        if not C_e:  # Handle case with no edges
            return torch.zeros_like(A)
            
        C_e = torch.stack(C_e)  # [num_edges, feat_dim]

        # Step 3: Compute edge embeddings
        y = self.mlp(C_e)  # [num_edges, 1]

        # Step 4: Gumbel-TopK sampling
        num_edges = len(y)
        num_keep = max(1, int(self.k_ratio * num_edges))  # Keep at least 1 edge

        # Add Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(y) + 1e-8) + 1e-8)
        perturbed_logits = y + gumbel_noise

        # Select Top-K edges
        _, topk_indices = torch.topk(perturbed_logits.squeeze(),
                                     k=num_keep, sorted=False)

        # Step 5: Build new adjacency matrix
        A_aug = torch.zeros_like(A)
        for idx in topk_indices:
            i, j = edge_positions[idx]
            A_aug[i, j] = A[i, j]  # Keep original weights

        return A_aug  # Augmented adjacency matrix


class GNNEncoder(nn.Module):
    """Graph Neural Network encoder with gradient stability"""
    def __init__(self, input_dim, hidden_dim, num_node_types=5, num_edge_types=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        
        # Node type embedding with small initialization
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim // 4)
        nn.init.normal_(self.node_type_embedding.weight, 0, 0.01)
        
        # Feature projection layer with batch normalization
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Message passing layer
        self.message_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh()  # Use Tanh to limit output range
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Weight initialization for gradient stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, A, X, node_types=None, edge_types=None):
        """
        A: adjacency matrix [n, n]
        X: node features [n, input_dim]
        node_types: node types [n] (optional)
        edge_types: edge type information (optional)
        """
        n = A.size(0)
        device = A.device
        
        # Ensure numerical stability
        X = torch.clamp(X, -10, 10)  # Limit input range
        A = torch.clamp(A, 0, 1)     # Ensure adjacency matrix in [0,1] range
        
        # Initial feature projection
        H = self.feature_proj(X)  # [n, hidden_dim]
        
        # Add node type information
        if node_types is not None:
            type_emb = self.node_type_embedding(node_types.to(device))  # [n, hidden_dim//4]
            # Zero padding instead of addition
            type_emb_padded = F.pad(type_emb, (0, self.hidden_dim - type_emb.size(1)))
            H = H + 0.1 * type_emb_padded  # Small weight for type information
        
        # Message passing with degree normalization
        degrees = torch.sum(A, dim=1, keepdim=True) + 1e-8
        A_norm = A / degrees
        
        # Message passing
        messages = self.message_proj(H)  # [n, hidden_dim]
        aggregated = torch.mm(A_norm, messages)  # [n, hidden_dim]
        
        # Residual connection
        Z = H + aggregated
        
        # Output projection
        Z = self.output_proj(Z)  # [n, hidden_dim]
        
        # Graph-level summary
        h_G = torch.mean(Z, dim=0, keepdim=True)  # [1, hidden_dim]
        
        return Z, h_G


# ===================== Loss Functions =====================
def subgraph_contrastive_loss(Z_orig, Z_aug, A_batch, center_indices, tau=0.2):
    """
    Subgraph-level contrastive loss function
    """
    if len(center_indices) == 0:
        return torch.tensor(0.0, requires_grad=True, device=Z_orig.device)
    
    # Limit embedding range
    Z_orig = torch.clamp(Z_orig, -5, 5)
    Z_aug = torch.clamp(Z_aug, -5, 5)
    
    # Compute neighbor aggregation with degree normalization
    degrees = torch.sum(A_batch, dim=1, keepdim=True) + 1e-8
    A_norm = A_batch / degrees
    
    neighbor_sum_orig = torch.mm(A_norm, Z_orig)  # [N, d]
    neighbor_sum_aug = torch.mm(A_norm, Z_aug)  # [N, d]

    # Extract center node features
    center_orig = Z_orig[center_indices]  # [B, d]
    center_aug = Z_aug[center_indices]  # [B, d]

    # Build subgraph summaries
    Z_summary_orig = center_orig + 0.5 * neighbor_sum_orig[center_indices]  # [B, d]
    Z_summary_aug = center_aug + 0.5 * neighbor_sum_aug[center_indices]  # [B, d]

    # Vector normalization
    Z_summary_orig = F.normalize(Z_summary_orig, p=2, dim=1)
    Z_summary_aug = F.normalize(Z_summary_aug, p=2, dim=1)

    # Compute similarity matrix
    logits = torch.mm(Z_summary_orig, Z_summary_aug.t()) / tau
    logits = torch.clamp(logits, -10, 10)  # Limit logits range

    # Build labels
    labels = torch.arange(Z_summary_orig.size(0), device=logits.device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss


def graph_contrastive_loss(Z_orig, Z_neg, tau=0.2):
    """
    Graph-level contrastive loss
    """
    # Limit embedding range
    Z_orig = torch.clamp(Z_orig, -5, 5)
    Z_neg = torch.clamp(Z_neg, -5, 5)
    
    # Graph summary
    z_g = Z_orig.mean(dim=0, keepdim=True)  # [1, d]
    z_g = F.normalize(z_g, p=2, dim=1)

    # Positive sample scores
    Z_orig_norm = F.normalize(Z_orig, p=2, dim=1)
    pos_scores = torch.sum(Z_orig_norm * z_g, dim=1) / tau  # [N]
    
    # Negative sample scores
    Z_neg_norm = F.normalize(Z_neg, p=2, dim=1)
    neg_scores = torch.sum(Z_neg_norm * z_g, dim=1) / tau  # [M]
    
    # Limit score range
    pos_scores = torch.clamp(pos_scores, -10, 10)
    neg_scores = torch.clamp(neg_scores, -10, 10)

    # Contrastive loss
    pos_loss = F.softplus(-pos_scores).mean()
    neg_loss = F.softplus(neg_scores).mean()
    
    loss = (pos_loss + neg_loss) / 2
    return torch.clamp(loss, 0, 10)  # Limit loss range


def node_contrastive_loss(Z, X, X_neg, proj_head, tau=0.2):
    """
    Node-level contrastive loss
    """
    # Limit input range
    Z = torch.clamp(Z, -5, 5)
    X = torch.clamp(X, -10, 10)
    X_neg = torch.clamp(X_neg, -10, 10)
    
    # Project features
    X_proj = proj_head(X)  # [N, d]
    X_neg_proj = proj_head(X_neg)  # [M, d]

    # Normalization
    Z = F.normalize(Z, p=2, dim=1)
    X_proj = F.normalize(X_proj, p=2, dim=1)
    X_neg_proj = F.normalize(X_neg_proj, p=2, dim=1)

    # Positive sample scores
    pos_scores = torch.sum(Z * X_proj, dim=1) / tau  # [N]

    # Negative sample scores
    neg_scores = torch.sum(Z * X_neg_proj, dim=1) / tau  # [N]

    # Limit score range
    pos_scores = torch.clamp(pos_scores, -10, 10)
    neg_scores = torch.clamp(neg_scores, -10, 10)

    # Contrastive loss
    pos_loss = F.softplus(-pos_scores).mean()
    neg_loss = F.softplus(neg_scores).mean()
    
    loss = (pos_loss + neg_loss) / 2
    return torch.clamp(loss, 0, 10)


class MSCIA(nn.Module):
    """Multi-Scale Contrastive InfoMax Augmentation model"""
    def __init__(self, input_dim, hidden_dim, k_ratio, alpha, beta, gamma, tau=0.2,
                 augment_type='edge', num_node_types=5, num_edge_types=3):
        super().__init__()
        
        # Graph encoders
        self.aug_gnn = GNNEncoder(input_dim, hidden_dim, num_node_types, num_edge_types)
        self.encoder = GNNEncoder(input_dim, hidden_dim, num_node_types, num_edge_types)
        
        # MLPs for augmentation
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Limit output range
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Limit output range
        )

        self.tau = tau
        self.proj_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Limit output range
        )

        # Select augmentation type
        self.augment_type = augment_type
        if augment_type == 'edge':
            self.augmentor = EdgeSamplingHead(self.aug_gnn, self.edge_mlp, k_ratio)
        elif augment_type == 'node':
            self.augmentor = NodeSamplingHead(self.aug_gnn, self.node_mlp, k_ratio)
        else:
            raise ValueError(f"Unknown augment type: {augment_type}")

        # Loss weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, A_batch, X_batch, center_indices, node_types=None, edge_types=None):
        """
        A_batch: adjacency matrix [N, N]
        X_batch: node features [N, F]
        center_indices: center node indices list [B]
        node_types: node types [N] (optional)
        edge_types: edge type information (optional)
        """
        device = A_batch.device
        
        # 1. Graph augmentation
        try:
            A_aug = self.augmentor(A_batch, X_batch)
        except Exception as e:
            print(f"Graph augmentation failed: {e}")
            A_aug = A_batch.clone()  # Use original graph as fallback

        # 2. Negative sample generation (feature permutation)
        idx = torch.randperm(X_batch.size(0), device=device)
        X_neg = X_batch[idx]

        # 3. Shared encoder
        try:
            Z_orig, _ = self.encoder(A_batch, X_batch, node_types, edge_types)  # [N, d]
            Z_aug, _ = self.encoder(A_aug, X_batch, node_types, edge_types)  # [N, d]
            Z_neg, _ = self.encoder(A_batch, X_neg, node_types, edge_types)  # [N, d]
        except Exception as e:
            print(f"Encoder failed: {e}")
            return torch.tensor(float('inf'), requires_grad=True, device=device), X_batch

        # 4. Multi-scale contrastive loss
        try:
            loss_subgraph = subgraph_contrastive_loss(
                Z_orig, Z_aug, A_batch, center_indices, tau=self.tau
            )
            loss_graph = graph_contrastive_loss(Z_orig, Z_neg, tau=self.tau)
            loss_node = node_contrastive_loss(Z_orig, X_batch, X_neg, self.proj_head, tau=self.tau)
        except Exception as e:
            print(f"Loss computation failed: {e}")
            return torch.tensor(float('inf'), requires_grad=True, device=device), Z_orig

        # 5. Weighted total loss
        total_loss = (self.alpha * loss_graph +
                      self.beta * loss_node +
                      self.gamma * loss_subgraph)
        
        total_loss = torch.clamp(total_loss, 0, 20)  # Limit total loss range

        return total_loss, Z_orig


# ===================== Dataset Class =====================
class BlockchainFeatureDataset(Dataset):
    """Blockchain network feature dataset"""
    def __init__(self, classic_feature_file, adjacency_file, time_window=1, batch_size=32):
        # Load classic features
        self.classic_df = pd.read_csv(classic_feature_file)
        print(f"Classic feature data shape: {self.classic_df.shape}")

        # Ensure first column is node_id
        if 'node_id' not in self.classic_df.columns:
            self.classic_df = self.classic_df.rename(columns={self.classic_df.columns[0]: 'node_id'})

        # Extract feature columns
        self.feature_columns = [col for col in self.classic_df.columns
                               if col.startswith('f_classic_')]
        print(f"Number of classic feature columns: {len(self.feature_columns)}")

        # Load adjacency matrix and graph structure information
        self.graph_data = torch.load(adjacency_file, map_location='cpu')
        print(f"Graph data loaded, number of nodes: {self.graph_data['metadata']['num_nodes']}")

        # Extract graph structure information
        self.adjacency_matrix = self.graph_data['adjacency_matrix']  # [200, 200]
        self.edge_index = self.graph_data['edge_index']  # [2, 4485]
        self.edge_type = self.graph_data['edge_type']  # [4485]
        self.node_types = self.graph_data['metadata']['node_types']  # List[str]
        
        # Node type encoding
        self.node_type_mapping = {
            'miner': 0, 'full_node': 1, 'light_node': 2,
            'validator': 3, 'storage': 4
        }
        self.node_type_tensor = torch.tensor([
            self.node_type_mapping.get(nt, 0) for nt in self.node_types
        ], dtype=torch.long)

        # Extract and standardize features
        self.features = self.classic_df[self.feature_columns].values
        self.node_ids = self.classic_df['node_id'].values
        
        # Robust feature standardization
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        # Limit feature range to prevent extreme values
        self.scaled_features = np.clip(self.scaled_features, -5, 5)

        # Create virtual timestamps (since this is a static graph)
        # Enhanced to support real timestamps from BlockEmulator
        if 'timestamp' not in self.classic_df.columns:
            self.classic_df['timestamp'] = 0
            print("Created virtual timestamps, treating graph as static snapshot")
        else:
            unique_timestamps = self.classic_df['timestamp'].unique()
            if len(unique_timestamps) > 1:
                print(f"Using real timestamps: {min(unique_timestamps)} - {max(unique_timestamps)}")
            else:
                print("Single timestamp detected, treating as static snapshot")

        self.timestamps = sorted(self.classic_df['timestamp'].unique())
        self.time_window = time_window
        self.batch_size = batch_size

        print(f"Dataset initialization complete:")
        print(f"- Number of nodes: {len(self.node_ids)}")
        print(f"- Feature dimension: {len(self.feature_columns)}")
        print(f"- Number of edges: {self.edge_index.size(1)}")
        print(f"- Number of node types: {len(set(self.node_types))}")

    def __len__(self):
        return max(1, len(self.timestamps))  # Return at least 1

    def __getitem__(self, idx):
        """Get graph data batch with real timestamp support"""
        # Enhanced timestamp handling
        if len(self.timestamps) > 1:
            # Use different timestamps for different batches when available
            ts_idx = idx % len(self.timestamps)
            ts = self.timestamps[ts_idx]
            
            # Filter data for current timestamp
            current_data = self.classic_df[self.classic_df['timestamp'] == ts]
            if len(current_data) == 0:
                # Fallback to first timestamp
                ts = self.timestamps[0]
                current_data = self.classic_df[self.classic_df['timestamp'] == ts]
        else:
            # Use first (and only) timestamp
            ts = self.timestamps[0]
            current_data = self.classic_df
        
        # Get node features for current timestamp
        if len(current_data) == len(self.classic_df):
            # All data belongs to this timestamp
            features = torch.tensor(self.scaled_features, dtype=torch.float)
            current_node_ids = self.node_ids
        else:
            # Filter features for current timestamp
            feature_indices = current_data.index
            current_features = self.scaled_features[feature_indices]
            features = torch.tensor(current_features, dtype=torch.float)
            current_node_ids = current_data['node_id'].values
        
        # Get adjacency matrix (assume it's static for now)
        adj_matrix = self.adjacency_matrix.clone().float()
        
        # Adjust adjacency matrix size if needed
        num_nodes = features.size(0)
        if adj_matrix.size(0) != num_nodes:
            # Resize or recreate adjacency matrix
            if num_nodes <= adj_matrix.size(0):
                adj_matrix = adj_matrix[:num_nodes, :num_nodes]
            else:
                # Expand with zeros
                new_adj = torch.zeros(num_nodes, num_nodes)
                min_size = min(adj_matrix.size(0), num_nodes)
                new_adj[:min_size, :min_size] = adj_matrix[:min_size, :min_size]
                adj_matrix = new_adj
        
        # Get node types
        if len(current_data) == len(self.node_types):
            node_types = self.node_type_tensor.clone()[:num_nodes]
        else:
            # Generate node types for current nodes
            node_types = torch.zeros(num_nodes, dtype=torch.long)
            for i, node_id in enumerate(current_node_ids):
                # Use modulo to assign types cyclically
                node_types[i] = node_id % len(self.node_type_mapping)
        
        # Randomly select center nodes
        available_nodes = list(range(num_nodes))
        
        # Select batch_size center nodes
        num_centers = min(self.batch_size, num_nodes)
        center_indices = torch.randperm(num_nodes)[:num_centers].tolist()
        selected_nodes = [current_node_ids[i] for i in center_indices]
        
        return {
            'adjacency_matrix': adj_matrix,
            'node_features': features,
            'center_indices': center_indices,
            'selected_nodes': selected_nodes,
            'node_types': node_types,
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'timestamp': ts
        }


# ===================== Temporal MS-CIA Model =====================
class TemporalMSCIA(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_dim, k_ratio, alpha, beta, gamma,
                 tau=0.2, num_node_types=5, num_edge_types=3):
        super().__init__()
        
        # Time embedding
        self.time_embedding = nn.Embedding(1000, time_dim)
        nn.init.normal_(self.time_embedding.weight, 0, 0.01)  # Small initialization
        
        # Feature projection (fusing time information)
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # MS-CIA core
        self.mscia = MSCIA(
            hidden_dim, hidden_dim, k_ratio, alpha, beta, gamma, tau,
            num_node_types=num_node_types, num_edge_types=num_edge_types
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh()  # Limit output range
        )
        
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.tau = tau

    def forward(self, batch_data):
        """
        batch_data: dictionary containing all necessary information
        """
        A_batch = batch_data['adjacency_matrix']
        X_batch = batch_data['node_features']
        center_indices = batch_data['center_indices']
        node_types = batch_data['node_types']
        timestamp = batch_data['timestamp']
        
        device = A_batch.device
        
        # Create timestamp tensor with enhanced handling
        num_nodes = X_batch.size(0)
        
        # Ensure timestamp is within embedding range
        max_embedding_ts = self.time_embedding.num_embeddings - 1
        if isinstance(timestamp, torch.Tensor):
            timestamp = timestamp.item()
        
        # Clamp timestamp to valid range
        timestamp = max(0, min(timestamp, max_embedding_ts))
        timestamps = torch.full((num_nodes,), timestamp, dtype=torch.long, device=device)
        
        # Time embedding with error handling
        try:
            time_emb = self.time_embedding(timestamps)  # [num_nodes, time_dim]
        except Exception as e:
            print(f"Time embedding error: {e}, using timestamp {timestamp}")
            # Fallback to zero timestamp
            timestamps = torch.zeros((num_nodes,), dtype=torch.long, device=device)
            time_emb = self.time_embedding(timestamps)
        
        # Fuse features and time information
        X_time = torch.cat([X_batch, time_emb], dim=1)  # [num_nodes, input_dim + time_dim]
        X_proj = self.feature_proj(X_time)  # [num_nodes, hidden_dim]

        # MS-CIA forward pass
        loss, embeddings = self.mscia(
            A_batch, X_proj, center_indices,
            node_types=node_types, edge_types=None
        )

        # Output projection
        output_emb = self.output_proj(embeddings)
        
        return loss, output_emb

    def get_temporal_embeddings(self, embeddings, timestamps, node_ids):
        """Organize temporal embeddings of nodes"""
        temporal_embeddings = {}

        for i, node_id in enumerate(node_ids):
            ts = timestamps[i].item() if torch.is_tensor(timestamps[i]) else timestamps[i]

            if node_id not in temporal_embeddings:
                temporal_embeddings[node_id] = {}

            temporal_embeddings[node_id][ts] = embeddings[i].detach().cpu().numpy()

        return temporal_embeddings


# ===================== Training Function =====================
def train_mscia(classic_feature_file, adjacency_file, config, external_timestamps=None):
    """Train MS-CIA model with fixed learning rate and real timestamp support"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Enhanced dataset with real timestamp support
    dataset = BlockchainFeatureDataset(
        classic_feature_file=classic_feature_file,
        adjacency_file=adjacency_file,
        time_window=config['time_window'],
        batch_size=config['batch_size']
    )
    
    # If external timestamps provided, update dataset
    if external_timestamps is not None:
        print(f"Updating dataset with {len(external_timestamps)} external timestamps")
        dataset.classic_df['timestamp'] = external_timestamps[:len(dataset.classic_df)]
        dataset.timestamps = sorted(dataset.classic_df['timestamp'].unique())
        print(f"Updated timestamps: {min(dataset.timestamps)} - {max(dataset.timestamps)}")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Enhanced model with larger timestamp embedding range
    max_timestamp = max(dataset.timestamps) if dataset.timestamps else 1000
    enhanced_time_embedding_size = max(1000, max_timestamp + 100)
    
    model = TemporalMSCIA(
        input_dim=len(dataset.feature_columns),
        hidden_dim=config['hidden_dim'],
        time_dim=config['time_dim'],
        k_ratio=config['k_ratio'],
        alpha=config['alpha'],
        beta=config['beta'],
        gamma=config['gamma'],
        tau=config['tau'],
        num_node_types=5,
        num_edge_types=3
    ).to(device)
    
    # Expand time embedding if needed
    if enhanced_time_embedding_size > model.time_embedding.num_embeddings:
        print(f"Expanding time embedding from {model.time_embedding.num_embeddings} to {enhanced_time_embedding_size}")
        old_embedding = model.time_embedding.weight.data
        model.time_embedding = nn.Embedding(enhanced_time_embedding_size, config['time_dim']).to(device)
        nn.init.normal_(model.time_embedding.weight, 0, 0.01)
        # Copy old weights
        model.time_embedding.weight.data[:old_embedding.size(0)] = old_embedding

    # Fixed learning rate with conservative optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],  # Fixed learning rate, no decay
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler (only triggered when loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=50, min_lr=1e-5
    )
    
    # Early stopping mechanism
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 40  # Moderate patience value
    best_model_state = None
    
    loss_history = []

    print("Starting training with fixed learning rate...")
    
    for epoch in range(config['epochs']):
        epoch_loss = 0.0
        valid_batches = 0

        for batch_idx, batch_data in enumerate(dataloader):
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].squeeze(0).to(device)
                elif isinstance(batch_data[key], list) and len(batch_data[key]) == 1:
                    batch_data[key] = batch_data[key][0]

            optimizer.zero_grad()
            
            try:
                loss, embeddings = model(batch_data)
                
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 20:
                    continue

                loss.backward()
                
                # Fixed gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                epoch_loss += loss.item()
                valid_batches += 1
                
            except Exception as e:
                continue

        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            loss_history.append(avg_loss)
            
            # Update learning rate scheduler (rarely triggered)
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Simple early stopping logic
            if avg_loss < best_loss - 1e-5:  # Simple improvement threshold
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss
                }
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{config['epochs']} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Best: {best_loss:.4f} | "
                      f"LR: {current_lr:.6f} (fixed) | "
                      f"Patience: {patience_counter}/{max_patience}")
            
            if patience_counter >= max_patience:
                print(f"Early stopping at Epoch {epoch + 1}, best loss: {best_loss:.4f}")
                if best_model_state:
                    model.load_state_dict(best_model_state['model_state_dict'])
                    print(f"Restored model from epoch {best_model_state['epoch']}")
                break

    print("Training completed!")
    return model, {}, loss_history


def generate_final_embeddings(model, classic_feature_file, adjacency_file, config):
    """Generate final node embeddings"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset
    dataset = BlockchainFeatureDataset(
        classic_feature_file=classic_feature_file,
        adjacency_file=adjacency_file,
        time_window=config['time_window'],
        batch_size=config['batch_size']
    )

    # Switch to evaluation mode
    model.eval()
    
    all_embeddings = {}

    with torch.no_grad():
        # Get single batch data (containing all nodes)
        batch_data = dataset[0]
        
        # Move data to device
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(device)
        
        # Modify center_indices to include all nodes
        num_nodes = len(dataset.node_ids)
        batch_data['center_indices'] = list(range(num_nodes))
        batch_data['selected_nodes'] = dataset.node_ids.tolist()

        # Generate embeddings
        _, embeddings = model(batch_data)

        # Save all node embeddings
        timestamps = [batch_data['timestamp']] * len(batch_data['selected_nodes'])
        temporal_embeddings = model.get_temporal_embeddings(
            embeddings, timestamps, batch_data['selected_nodes']
        )
        
        all_embeddings.update(temporal_embeddings)

    return all_embeddings


def visualize_embeddings(temporal_embeddings, output_dir="visualizations"):
    """Visualize temporal embedding results"""
    import os
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib as mpl
    
    # Add Chinese font support
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    mpl.rcParams['axes.unicode_minus'] = False

    os.makedirs(output_dir, exist_ok=True)

    # Collect latest embeddings for all nodes
    all_embeddings = []
    all_labels = []

    for node_id, time_emb in temporal_embeddings.items():
        if time_emb:
            # Take embedding from latest timestep
            latest_ts = max(time_emb.keys())
            all_embeddings.append(time_emb[latest_ts])
            all_labels.append(str(node_id))

    n_samples = len(all_embeddings)
    print(f"Visualizing {n_samples} node embeddings")

    if n_samples > 1:
        embeddings_array = np.array(all_embeddings)
        
        # Use t-SNE for dimensionality reduction
        if n_samples > 30:
            perplexity = min(30, max(5, n_samples // 3))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings_array)
            method = "t-SNE"
        else:
            # Use PCA when insufficient samples
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings_array)
            method = "PCA"

        # Create visualization
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            alpha=0.7, s=50, c=range(n_samples), cmap='tab10')

        # Annotate some node IDs
        for i in range(min(20, n_samples)):
            plt.annotate(all_labels[i],
                        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)

        plt.title(f"Blockchain Node Embedding Distribution ({method})", fontsize=14)
        plt.xlabel(f"{method} First Principal Component", fontsize=12)
        plt.ylabel(f"{method} Second Principal Component", fontsize=12)
        plt.colorbar(scatter, label='Node Index')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'blockchain_embeddings_{method.lower()}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Embedding visualization saved to {output_dir}/blockchain_embeddings_{method.lower()}.png")
        
        # Clustering visualization
        if n_samples > 20:
            try:
                from sklearn.cluster import KMeans
                
                # K-means clustering
                n_clusters = min(5, n_samples // 4)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings_array)
                
                # Clustering visualization
                plt.figure(figsize=(12, 10))
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                    c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
                
                plt.title(f"Blockchain Node Embedding Clustering (K={n_clusters})", fontsize=14)
                plt.xlabel(f"{method} First Principal Component", fontsize=12)
                plt.ylabel(f"{method} Second Principal Component", fontsize=12)
                plt.colorbar(scatter, label='Cluster Label')
                plt.grid(True, alpha=0.3)
                
                plt.savefig(os.path.join(output_dir, f'blockchain_embeddings_clusters.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Clustering visualization saved to {output_dir}/blockchain_embeddings_clusters.png")
            except ImportError:
                print("sklearn.cluster not available, skipping clustering visualization")
    else:
        print(f"Insufficient samples ({n_samples}), cannot visualize")


# ===================== Main Program =====================
if __name__ == "__main__":
    # Enhanced configuration with real timestamp support
    config = {
        'classic_feature_file': '../partition/feature/step1_large_samples_f_classic.csv',
        'adjacency_file': '../partition/feature/step1_adjacency_raw.pt',
        'time_window': 1,
        'batch_size': 32,
        'hidden_dim': 64,
        'time_dim': 16,
        'k_ratio': 0.9,              # Higher sampling ratio
        'alpha': 0.3,                # Lower graph-level loss weight
        'beta': 0.4,                 # Higher node-level loss weight  
        'gamma': 0.3,                # Maintain subgraph loss weight
        'lr': 0.02,                  # Moderate fixed learning rate
        'weight_decay': 9e-6,        # Smaller weight decay
        'tau': 0.09,                 # Lower temperature parameter
        'epochs': 300,               # More epochs for full training
        
        # Real timestamp support
        'use_real_timestamps': True,
        'max_timestamp': 100000,     # Support larger timestamp range
        'external_timestamps': None   # Can be provided for real data
    }
    
    print("=" * 70)
    print("Blockchain Multi-Scale Contrastive Learning Model")
    print("=" * 70)
    print(f"Strategy: Fixed learning rate {config['lr']}")
    print(f"Temperature: {config['tau']} (enhanced contrast)")
    print(f"Target: Loss below 0.25")
    print("=" * 70)

    try:
        trained_model, temporal_embeddings, loss_history = train_mscia(
            config['classic_feature_file'],
            config['adjacency_file'],
            config
        )

        print("Generating final embeddings...")
        final_embeddings = generate_final_embeddings(
            trained_model,
            config['classic_feature_file'],
            config['adjacency_file'],
            config
        )

        import pickle
        
        # Save results
        output_file = 'temporal_embeddings.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(final_embeddings, f)
        print(f"Embeddings saved to {output_file}")

        history_file = 'training_history.pkl'
        with open(history_file, 'wb') as f:
            pickle.dump({
                'loss_history': loss_history,
                'config': config
            }, f)
        print(f"Training history saved to {history_file}")

        model_file = 'mscia_model.pth'
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': config
        }, model_file)
        print(f"Model saved to {model_file}")

        print("Generating visualizations...")
        visualize_embeddings(final_embeddings)

        # Training summary
        print("\n" + "=" * 70)
        print("Training Summary:")
        if loss_history:
            final_loss = loss_history[-1]
            min_loss = min(loss_history)
            # Check if loss is still decreasing
            recent_trend = "increasing" if loss_history[-1] > loss_history[-10] else "decreasing"
            print(f"Final loss: {final_loss:.4f}")
            print(f"Best loss: {min_loss:.4f}")
            print(f"Recent trend: {recent_trend}")
            print(f"Target achieved: {'Yes' if min_loss < 0.25 else 'Continue training'}")
        print(f"Number of node embeddings: {len(final_embeddings)}")
        print(f"Embedding dimension: {config['hidden_dim']}")
        print("=" * 70)
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()