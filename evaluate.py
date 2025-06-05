import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from typing import Dict, Any, List
from preprocess import GraphDataset

logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate_model(model: nn.Module, 
                  dataset: GraphDataset,
                  device: torch.device,
                  batch_size: int = 32) -> Dict[str, Any]:
    """Comprehensive model evaluation with various metrics
    
    Args:
        model: Trained model
        dataset: Dataset for evaluation
        device: Computation device
        batch_size: Batch size
        
    Returns:
        Dict[str, Any]: Various evaluation metrics
    """
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    # Predictions for all samples
    num_batches = len(dataset) // batch_size + 1
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(dataset))
        
        batch_features = dataset.features[start_idx:end_idx].to(device)
        batch_labels = dataset.labels[start_idx:end_idx].to(device)
        batch_edge_indices = [edge_index.to(device) for edge_index in 
                          dataset.edge_indices[start_idx:end_idx]]
        
        outputs = []
        for features, edge_index in zip(batch_features, batch_edge_indices):
            # Ensure correct tensor shape
            if features.dim() == 1:
                features = features.unsqueeze(0)  # Convert to (1 × F)
            output = model(features, edge_index)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0)
        
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'auc_roc': roc_auc_score(all_labels, all_probs),
        'auc_pr': average_precision_score(all_labels, all_probs)
    }
    
    # Display results
    logger.info("\nEvaluation Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1']:.4f}")
    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"AUC-PR: {metrics['auc_pr']:.4f}")
    
    return metrics

def analyze_errors(model: nn.Module,
                  dataset: GraphDataset,
                  device: torch.device,
                  feature_names: List[str]) -> Dict[str, Any]:
    """Analyze model errors
    
    Args:
        model: Trained model
        dataset: Dataset
        device: Computation device
        feature_names: Feature names
        
    Returns:
        Dict[str, Any]: Error analysis results
    """
    model.eval()
    
    errors = {
        'false_positives': [],
        'false_negatives': [],
        'feature_importance': np.zeros(len(feature_names))
    }
    
    with torch.no_grad():
        for i in range(len(dataset)):
            features = dataset.features[i].to(device)
            label = dataset.labels[i].item()
            edge_index = dataset.edge_indices[i].to(device)
            
            # Ensure correct tensor shape
            if features.dim() == 1:
                features = features.unsqueeze(0)  # Convert to (1 × F)
            
            # Prediction
            output = model(features, edge_index)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            
            # Store errors
            if pred != label:
                error_info = {
                    'true_label': label,
                    'predicted': pred,
                    'confidence': prob[0, pred].item(),
                    'features': features.cpu().numpy()
                }
                
                if pred == 1 and label == 0:  # False Positive
                    errors['false_positives'].append(error_info)
                else:  # False Negative
                    errors['false_negatives'].append(error_info)
    
    # Analyze feature importance in errors
    for error_list in [errors['false_positives'], errors['false_negatives']]:
        for error in error_list:
            errors['feature_importance'] += np.abs(error['features'])
    
    if len(errors['false_positives']) + len(errors['false_negatives']) > 0:
        errors['feature_importance'] /= (
            len(errors['false_positives']) + len(errors['false_negatives'])
        )
    
    # Display results
    logger.info("\nError Analysis:")
    logger.info(f"Number of False Positives: {len(errors['false_positives'])}")
    logger.info(f"Number of False Negatives: {len(errors['false_negatives'])}")
    
    logger.info("\nTop Features in Errors:")
    feature_importance = list(zip(feature_names, errors['feature_importance']))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    for feature, importance in feature_importance[:5]:
        logger.info(f"{feature}: {importance:.4f}")
    
    return errors 