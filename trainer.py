import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Optional
from tqdm import tqdm

from preprocess import GraphDataset

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Model training management class"""
    
    def __init__(
        self,
        model: nn.Module,
        train_data: GraphDataset,
        val_data: GraphDataset,
        test_data: GraphDataset,
        device: torch.device,
        batch_size: int = 32,
        epochs: int = 150,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 15,
        **kwargs
    ):
        """
        Args:
            model: GNN model
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            device: Computation device (CPU/GPU)
            batch_size: Batch size
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Regularization coefficient
            patience: Number of epochs to wait for early stopping
        """
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=patience // 2
        )
        
        # Loss criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch
        
        Returns:
            Dict[str, float]: Evaluation metrics for the epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Split data into smaller batches
        num_batches = len(self.train_data) // self.batch_size + 1
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(self.train_data))
            
            batch_features = self.train_data.features[start_idx:end_idx]
            batch_labels = self.train_data.labels[start_idx:end_idx]
            batch_edge_indices = self.train_data.edge_indices[start_idx:end_idx]
            
            self.optimizer.zero_grad()
            
            outputs = []
            for features, edge_index in zip(batch_features, batch_edge_indices):
                # Ensure correct tensor shape
                if features.dim() == 1:
                    features = features.unsqueeze(0)  # Convert to (1 × F)
                output = self.model(features, edge_index)
                outputs.append(output)
            
            # Aggregate outputs
            outputs = torch.cat(outputs, dim=0)
            
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        return {
            'loss': total_loss / num_batches,
            'acc': 100. * correct / total
        }
    
    @torch.no_grad()
    def evaluate(self, dataset: GraphDataset) -> Dict[str, float]:
        """Evaluate model on a dataset
        
        Args:
            dataset: Dataset for evaluation
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        num_batches = len(dataset) // self.batch_size + 1
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(dataset))
            
            batch_features = dataset.features[start_idx:end_idx]
            batch_labels = dataset.labels[start_idx:end_idx]
            batch_edge_indices = dataset.edge_indices[start_idx:end_idx]
            
            outputs = []
            for features, edge_index in zip(batch_features, batch_edge_indices):
                # Ensure correct tensor shape
                if features.dim() == 1:
                    features = features.unsqueeze(0)  # Convert to (1 × F)
                output = self.model(features, edge_index)
                outputs.append(output)
            
            # Aggregate outputs
            outputs = torch.cat(outputs, dim=0)
            
            loss = self.criterion(outputs, batch_labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        return {
            'loss': total_loss / num_batches,
            'acc': 100. * correct / total
        }
    
    def train(self) -> nn.Module:
        """Complete model training
        
        Returns:
            nn.Module: Best model based on validation loss
        """
        logger.info("Starting training...")
        
        for epoch in range(self.epochs):
            # Training
            train_metrics = self.train_epoch()
            
            # Evaluation
            val_metrics = self.evaluate(self.val_data)
            test_metrics = self.evaluate(self.test_data)
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['test_acc'].append(test_metrics['acc'])
            
            # Display results
            logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Train Acc: {train_metrics['acc']:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val Acc: {val_metrics['acc']:.2f}% - "
                f"Test Loss: {test_metrics['loss']:.4f} - "
                f"Test Acc: {test_metrics['acc']:.2f}%"
            )
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_model_state = self.model.state_dict()
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Return best model
        self.model.load_state_dict(self.best_model_state)
        return self.model 