import logging
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data, Batch
from util import TweetNode

logger = logging.getLogger(__name__)

class GraphDataset:
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 adjacency_list: List[np.ndarray], feature_names: List[str]):
        """
        Args:
            features: Features of all nodes (N × F)
            labels: Sample labels
            adjacency_list: List of adjacency matrices
            feature_names: Feature names
        """
        # Convert features and labels to tensors
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
        # Convert adjacency matrices to edge_index
        self.edge_indices = []
        for adj in adjacency_list:
            if len(adj) > 0:
                # Convert edge list to edge_index
                edge_index = torch.tensor(adj, dtype=torch.long)
                # Ensure edge_index has correct shape (2 × E)
                if edge_index.dim() == 2:
                    if edge_index.size(0) != 2:
                        edge_index = edge_index.t()
                else:
                    # If edge_index is 1D, convert to 2D
                    edge_index = edge_index.view(2, -1)
                self.edge_indices.append(edge_index.contiguous())
            else:
                # For empty graphs
                self.edge_indices.append(torch.zeros((2, 0), dtype=torch.long))
        
        self.feature_names = feature_names
        self.num_features = features.shape[1]
    
    def to(self, device):
        """Transfer data to specified device"""
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
        self.edge_indices = [edge_index.to(device) for edge_index in self.edge_indices]
        return self
    
    def __len__(self):
        """Number of samples"""
        return len(self.labels)

def extract_features(tweet_node: TweetNode) -> Dict[str, float]:
    """Extract features from a tweet node"""
    features = {
        'bot_score': float(tweet_node.botometer_score if tweet_node.botometer_score else 0),
        'sentiment_pos': float(tweet_node.sentiment['positive'] if tweet_node.sentiment else 0),
        'sentiment_neg': float(tweet_node.sentiment['negative'] if tweet_node.sentiment else 0),
        'sentiment_neu': float(tweet_node.sentiment['neutral'] if tweet_node.sentiment else 0),
        'num_retweets': len(tweet_node.retweet_children),
        'num_replies': len(tweet_node.reply_children),
        'is_root': int(tweet_node.node_type == 'root'),
        'depth': 0  # will be updated in build_propagation_tree
    }
    return features

def build_propagation_tree(root_node: TweetNode) -> Tuple[List[Dict[str, float]], np.ndarray]:
    """Build propagation tree and extract features"""
    features = []
    edges = []
    node_to_idx = {}
    
    def dfs(node: TweetNode, depth: int = 0, parent_idx: int = None):
        if node.tweet_id in node_to_idx:
            return node_to_idx[node.tweet_id]
        
        curr_idx = len(features)
        node_to_idx[node.tweet_id] = curr_idx
        
        # Extract features
        node_features = extract_features(node)
        node_features['depth'] = depth
        features.append(node_features)
        
        # Add edge to graph
        if parent_idx is not None:
            edges.append([parent_idx, curr_idx])
        
        # Traverse children
        for child in node.retweet_children + node.reply_children:
            dfs(child, depth + 1, curr_idx)
            
        return curr_idx
    
    dfs(root_node)
    return features, np.array(edges)

def preprocess_samples(fake_samples: List[TweetNode], 
                      real_samples: List[TweetNode],
                      n_splits: int = 5) -> Tuple[GraphDataset, GraphDataset, GraphDataset]:
    """Preprocess samples and split into train/val/test"""
    logger.info("Extracting features and building propagation trees...")
    
    all_features = []
    all_edges = []
    all_labels = []
    
    # Process fake samples
    for sample in fake_samples:
        features, edges = build_propagation_tree(sample)
        all_features.append(features)
        all_edges.append(edges)
        all_labels.append(1)  # 1 for fake
        
    # Process real samples
    for sample in real_samples:
        features, edges = build_propagation_tree(sample)
        all_features.append(features)
        all_edges.append(edges)
        all_labels.append(0)  # 0 for real
    
    # Convert features to matrix
    feature_names = list(all_features[0][0].keys())
    features_matrix = np.array([[feat[name] for name in feature_names] 
                               for sample_features in all_features 
                               for feat in sample_features])
    
    # Normalize features
    logger.info("Normalizing features...")
    scaler = StandardScaler()
    features_matrix = scaler.fit_transform(features_matrix)
    
    # Split data
    logger.info("Splitting dataset...")
    train_idx, temp_idx = train_test_split(range(len(all_labels)), 
                                         test_size=0.3, 
                                         stratify=all_labels,
                                         random_state=42)
    val_idx, test_idx = train_test_split(temp_idx,
                                        test_size=0.5,
                                        stratify=[all_labels[i] for i in temp_idx],
                                        random_state=42)
    
    # Create datasets
    train_data = GraphDataset(
        features=features_matrix[train_idx],
        labels=np.array(all_labels)[train_idx],
        adjacency_list=[all_edges[i] for i in train_idx],
        feature_names=feature_names
    )
    
    val_data = GraphDataset(
        features=features_matrix[val_idx],
        labels=np.array(all_labels)[val_idx],
        adjacency_list=[all_edges[i] for i in val_idx],
        feature_names=feature_names
    )
    
    test_data = GraphDataset(
        features=features_matrix[test_idx],
        labels=np.array(all_labels)[test_idx],
        adjacency_list=[all_edges[i] for i in test_idx],
        feature_names=feature_names
    )
    
    return train_data, val_data, test_data 