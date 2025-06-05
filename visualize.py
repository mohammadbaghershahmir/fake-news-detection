import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from pathlib import Path
import networkx as nx
from util import TweetNode

logger = logging.getLogger(__name__)

def plot_training_history(history: Dict[str, List[float]], save_path: str) -> None:
    """رسم نمودار تاریخچه آموزش
    
    Args:
        history: تاریخچه معیارهای مختلف در طول آموزش
        save_path: مسیر ذخیره نمودار
    """
    plt.figure(figsize=(12, 4))
    
    # نمودار loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.plot(history['test_loss'], label='Test')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # نمودار accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training history plot saved to {save_path}")

def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """رسم ماتریس درهم‌ریختگی
    
    Args:
        cm: ماتریس درهم‌ریختگی
        save_path: مسیر ذخیره نمودار
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {save_path}")

def plot_feature_importance(model: Any, feature_names: List[str], 
                          save_path: str) -> None:
    """رسم نمودار اهمیت ویژگی‌ها
    
    Args:
        model: مدل آموزش دیده
        feature_names: نام ویژگی‌ها
        save_path: مسیر ذخیره نمودار
    """
    # استخراج وزن‌های لایه اول به عنوان معیار اهمیت
    weights = model.convs[0].weight.detach().cpu().numpy().mean(axis=0)
    importance = np.abs(weights)
    
    # مرتب‌سازی بر اساس اهمیت
    sorted_idx = np.argsort(importance)
    pos = np.arange(len(feature_names))
    
    plt.figure(figsize=(10, 6))
    plt.barh(pos, importance[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Analysis')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Feature importance plot saved to {save_path}")

def plot_propagation_patterns(fake_samples: List[TweetNode],
                            real_samples: List[TweetNode],
                            save_path: str) -> None:
    """تحلیل و نمایش الگوهای انتشار
    
    Args:
        fake_samples: نمونه‌های اخبار جعلی
        real_samples: نمونه‌های اخبار واقعی
        save_path: مسیر ذخیره نمودار
    """
    def get_graph_metrics(samples: List[TweetNode]) -> Dict[str, float]:
        """محاسبه معیارهای گراف برای یک مجموعه از نمونه‌ها"""
        metrics = {
            'avg_depth': [],
            'avg_branching': [],
            'avg_retweet_ratio': [],
            'avg_reply_ratio': []
        }
        
        for sample in samples:
            # تبدیل به گراف NetworkX
            G = nx.DiGraph()
            nodes_to_process = [(sample, None)]  # (node, parent)
            while nodes_to_process:
                node, parent = nodes_to_process.pop(0)
                G.add_node(node.tweet_id)
                if parent:
                    G.add_edge(parent.tweet_id, node.tweet_id)
                nodes_to_process.extend([(child, node) for child in 
                                       node.retweet_children + node.reply_children])
            
            # محاسبه معیارها
            depths = nx.shortest_path_length(G, source=sample.tweet_id)
            metrics['avg_depth'].append(max(depths.values()))
            
            branching = [len(list(G.successors(n))) for n in G.nodes()]
            metrics['avg_branching'].append(np.mean(branching) if branching else 0)
            
            total_children = len(sample.retweet_children) + len(sample.reply_children)
            if total_children > 0:
                metrics['avg_retweet_ratio'].append(
                    len(sample.retweet_children) / total_children
                )
                metrics['avg_reply_ratio'].append(
                    len(sample.reply_children) / total_children
                )
        
        return {k: np.mean(v) for k, v in metrics.items() if v}
    
    # محاسبه معیارها برای هر دو نوع خبر
    fake_metrics = get_graph_metrics(fake_samples)
    real_metrics = get_graph_metrics(real_samples)
    
    # رسم نمودار مقایسه‌ای
    metrics = list(fake_metrics.keys())
    fake_values = [fake_metrics[m] for m in metrics]
    real_values = [real_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, fake_values, width, label='Fake News')
    plt.bar(x + width/2, real_values, width, label='Real News')
    
    plt.xlabel('Metrics')
    plt.ylabel('Average Value')
    plt.title('Propagation Pattern Analysis')
    plt.xticks(x, [m.replace('avg_', '').replace('_', ' ').title() 
                   for m in metrics])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Propagation patterns plot saved to {save_path}")

def plot_attention_weights(attention_weights: List[np.ndarray],
                         feature_names: List[str],
                         save_path: str) -> None:
    """نمایش وزن‌های attention در لایه‌های مختلف
    
    Args:
        attention_weights: وزن‌های attention برای هر لایه
        feature_names: نام ویژگی‌ها
        save_path: مسیر ذخیره نمودار
    """
    n_layers = len(attention_weights)
    plt.figure(figsize=(15, 3 * n_layers))
    
    for i, weights in enumerate(attention_weights):
        plt.subplot(n_layers, 1, i + 1)
        sns.heatmap(weights, xticklabels=feature_names,
                   yticklabels=False, cmap='YlOrRd')
        plt.title(f'Layer {i+1} Attention Weights')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Attention weights plot saved to {save_path}")

def create_visualization_report(history: Dict[str, List[float]],
                             metrics: Dict[str, Any],
                             model: Any,
                             feature_names: List[str],
                             fake_samples: List[TweetNode],
                             real_samples: List[TweetNode],
                             save_dir: str) -> None:
    """ایجاد گزارش کامل تصویری
    
    Args:
        history: تاریخچه آموزش
        metrics: معیارهای ارزیابی
        model: مدل آموزش دیده
        feature_names: نام ویژگی‌ها
        fake_samples: نمونه‌های اخبار جعلی
        real_samples: نمونه‌های اخبار واقعی
        save_dir: دایرکتوری ذخیره نمودارها
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # رسم همه نمودارها
    plot_training_history(history, save_dir / 'training_history.png')
    plot_confusion_matrix(metrics['confusion_matrix'], 
                         save_dir / 'confusion_matrix.png')
    plot_feature_importance(model, feature_names,
                          save_dir / 'feature_importance.png')
    plot_propagation_patterns(fake_samples, real_samples,
                            save_dir / 'propagation_patterns.png')
    
    # اگر مدل از نوع GAT است، وزن‌های attention را هم نمایش بده
    if hasattr(model, 'get_attention_weights'):
        attention_weights = model.get_attention_weights()
        if attention_weights:
            plot_attention_weights(attention_weights, feature_names,
                                save_dir / 'attention_weights.png')
    
    logger.info(f"All visualization plots saved to {save_dir}") 