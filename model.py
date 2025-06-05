import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.loader import DataLoader
from typing import List, Dict, Optional, Tuple
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

class FakeNewsDetector(nn.Module):
    """Fake news detection model using GNN"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3,
                 dropout: float = 0.5, architecture: str = 'gcn'):
        """
        Args:
            input_dim: Input dimensions (number of features)
            hidden_dim: Hidden layer dimensions
            num_layers: Number of GNN layers
            dropout: Dropout rate
            architecture: Architecture type ('gat', 'gcn', or 'sage')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Select GNN layer type
        if architecture == 'gat':
            conv_layer = lambda in_dim, out_dim: GATConv(in_dim, out_dim, heads=8, concat=False)
        elif architecture == 'gcn':
            conv_layer = GCNConv
        elif architecture == 'sage':
            conv_layer = SAGEConv
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(conv_layer(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Last layer
        self.convs.append(conv_layer(hidden_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 2)  # 2 classes: real/fake
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize model weights"""
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix (N × F)
            edge_index: Edge list (2 × E)
            batch: Batch index for each node (N)
            
        Returns:
            logits: Class probabilities (B × 2)
        """
        # Apply GNN layers
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Last GNN layer
        x = self.convs[-1](x, edge_index)
        x = self.norms[-1](x)
        x = F.relu(x)
        
        # Global pooling
        if batch is not None:
            x = torch.cat([
                x[batch == i].mean(dim=0).unsqueeze(0)
                for i in range(batch.max().item() + 1)
            ], dim=0)
        else:
            x = x.mean(dim=0).unsqueeze(0)
        
        # Classification
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, 
                            edge_index: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention weights (only for GAT)
        
        Returns:
            List[torch.Tensor]: Attention weights for each layer
        """
        attention_weights = []
        
        for conv in self.convs:
            if isinstance(conv, GATConv):
                _, attention = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append(attention)
                x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)
        
        return attention_weights

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch.
    
    Args:
        model: The neural network model.
        loader: Data loader.
        optimizer: Optimizer.
        device: Device to run on.
        
    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Validate the model.
    
    Args:
        model: The neural network model.
        loader: Data loader.
        device: Device to run on.
        
    Returns:
        Tuple of (validation loss, metrics dictionary).
    """
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='binary'),
        'recall': recall_score(labels, predictions, average='binary'),
        'f1': f1_score(labels, predictions, average='binary')
    }
    
    return total_loss / len(loader), metrics

def train_with_cross_validation(
    data_list: List[Data],
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.5,
    architecture: str = 'gcn',
    n_splits: int = 5,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 10
) -> Tuple[nn.Module, Dict[str, float], Dict[str, List[float]]]:
    """Train and evaluate the model using k-fold cross-validation.
    
    Args:
        data_list: List of PyTorch Geometric Data objects.
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of graph convolution layers.
        dropout: Dropout rate.
        architecture: Model architecture ('gcn', 'gat', or 'sage').
        n_splits: Number of folds for cross-validation.
        batch_size: Batch size.
        epochs: Number of epochs.
        lr: Learning rate.
        weight_decay: Weight decay for regularization.
        patience: Patience for early stopping.
        
    Returns:
        Tuple of (best model, average metrics, history dictionary).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize metrics storage
    fold_metrics = []
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    
    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_list)):
        logger.info(f"Training fold {fold + 1}/{n_splits}")
        
        # Create data loaders
        train_loader = DataLoader([data_list[i] for i in train_idx], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=batch_size)
        
        # Initialize model and optimizer
        model = FakeNewsDetector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            architecture=architecture
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=patience)
        
        # Training loop
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_metrics = validate(model, val_loader, device)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            logger.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val Metrics: {val_metrics}")
            
            # Early stopping
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
        
        fold_metrics.append(val_metrics)
    
    # Calculate average metrics across folds
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        values = [m[metric] for m in fold_metrics]
        avg_metrics[metric] = np.mean(values)
        std = np.std(values)
        logger.info(f"Average {metric}: {avg_metrics[metric]:.4f} ± {std:.4f}")
    
    # Load best model
    best_model = FakeNewsDetector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        architecture=architecture
    ).to(device)
    best_model.load_state_dict(best_model_state)
    
    return best_model, avg_metrics, history

def predict(
    model: nn.Module,
    data: Data,
    device: torch.device
) -> Tuple[int, float]:
    """Make prediction for a single graph.
    
    Args:
        model: Trained model.
        data: PyTorch Geometric Data object.
        device: Device to run on.
        
    Returns:
        Tuple of (predicted class, confidence score).
    """
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        prob = torch.exp(out)
        pred_class = out.argmax(dim=1).item()
        confidence = prob[0][pred_class].item()
    return pred_class, confidence