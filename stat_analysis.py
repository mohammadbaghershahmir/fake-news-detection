import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_statistical_tests(
    fake_samples: np.ndarray,
    real_samples: np.ndarray
) -> Dict[str, Tuple[float, float]]:
    """Perform multiple statistical tests to compare fake and real samples.

    Args:
        fake_samples: Array of feature values for fake news.
        real_samples: Array of feature values for real news.

    Returns:
        Dictionary containing test results (test statistic and p-value) for each test.
    """
    results = {}
    
    # Student's t-test
    t_stat, t_pval = stats.ttest_ind(fake_samples, real_samples, equal_var=True)
    results['t_test'] = (t_stat, t_pval)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pval = stats.mannwhitneyu(fake_samples, real_samples, alternative='two-sided')
    results['mann_whitney'] = (u_stat, u_pval)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(fake_samples, real_samples)
    results['ks_test'] = (ks_stat, ks_pval)
    
    return results

def create_feature_distribution_plot(
    fake_samples: np.ndarray,
    real_samples: np.ndarray,
    feature_name: str,
    test_results: Dict[str, Tuple[float, float]],
    save_path: Path,
    figsize: tuple = (10, 6)
) -> None:
    """Create and save a comprehensive distribution plot comparing fake and real samples.

    Args:
        fake_samples: Array of feature values for fake news.
        real_samples: Array of feature values for real news.
        feature_name: Name of the feature being plotted.
        test_results: Dictionary of statistical test results.
        save_path: Path to save the plot.
        figsize: Figure size (width, height).
    """
    plt.figure(figsize=figsize)
    
    # Create subplot grid
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
    
    # Main distribution plot
    ax0 = plt.subplot(gs[0, :])
    sns.kdeplot(data=fake_samples, label='Fake', color='red', alpha=0.6)
    sns.kdeplot(data=real_samples, label='Real', color='blue', alpha=0.6)
    plt.title(f'Distribution of {feature_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Box plot
    ax1 = plt.subplot(gs[1, 0])
    data = pd.DataFrame({
        'Value': np.concatenate([fake_samples, real_samples]),
        'Label': ['Fake'] * len(fake_samples) + ['Real'] * len(real_samples)
    })
    sns.boxplot(x='Label', y='Value', data=data, palette={'Fake': 'red', 'Real': 'blue'})
    plt.title('Box Plot')
    
    # Statistical test results
    ax2 = plt.subplot(gs[1, 1])
    ax2.axis('off')
    test_text = "Statistical Tests:\n"
    for test_name, (stat, pval) in test_results.items():
        test_text += f"{test_name}:\n  stat={stat:.4f}, p={pval:.4e}\n"
        if pval < 0.05:
            test_text += "  (Significant)\n"
        else:
            test_text += "  (Not significant)\n"
    ax2.text(0, 0.5, test_text, fontsize=10, va='center')
    
    plt.tight_layout()
    plt.savefig(save_path / f"{feature_name.replace(' ', '_')}_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(
    features: np.ndarray,
    feature_names: List[str],
    save_path: Path,
    figsize: tuple = (12, 10)
) -> None:
    """Create and save a correlation heatmap for features.

    Args:
        features: Array of all features.
        feature_names: List of feature names.
        save_path: Path to save the plot.
        figsize: Figure size (width, height).
    """
    corr_matrix = np.corrcoef(features.T)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='.2f',
        square=True
    )
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dimensionality_reduction_plot(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    method: str = 'pca',
    figsize: tuple = (10, 8)
) -> None:
    """Create and save dimensionality reduction visualization.

    Args:
        features: Array of all features.
        labels: Array of labels (0 for fake, 1 for real).
        save_path: Path to save the plot.
        method: Dimensionality reduction method ('pca' or 'tsne').
        figsize: Figure size (width, height).
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA Visualization'
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE Visualization'
    
    reduced_features = reducer.fit_transform(features_scaled)
    
    # Create plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=labels,
        cmap='coolwarm',
        alpha=0.6
    )
    plt.colorbar(scatter)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.tight_layout()
    plt.savefig(save_path / f'{method.lower()}_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_features(
    fake_features: np.ndarray,
    real_features: np.ndarray,
    feature_names: List[str],
    save_folder: str = "plots"
) -> Dict[str, Dict[str, float]]:
    """Analyze and visualize features for fake and real news.

    Args:
        fake_features: Array of features for fake news.
        real_features: Array of features for real news.
        feature_names: List of feature names.
        save_folder: Directory to save the plots.

    Returns:
        Dictionary containing analysis results for each feature.
    """
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize results dictionary
    results = {}
    
    # Analyze each feature individually
    logger.info("Analyzing individual features...")
    for i, feature_name in enumerate(feature_names):
        logger.info(f"Analyzing feature: {feature_name}")
        
        # Perform statistical tests
        test_results = perform_statistical_tests(
            fake_features[:, i],
            real_features[:, i]
        )
        
        # Create distribution plot
        create_feature_distribution_plot(
            fake_features[:, i],
            real_features[:, i],
            feature_name,
            test_results,
            save_path
        )
        
        # Store results
        results[feature_name] = {
            f"{test_name}_pvalue": pval
            for test_name, (_, pval) in test_results.items()
        }
    
    # Create correlation heatmap
    logger.info("Creating correlation heatmap...")
    all_features = np.vstack([fake_features, real_features])
    create_correlation_heatmap(all_features, feature_names, save_path)
    
    # Create dimensionality reduction visualizations
    logger.info("Creating dimensionality reduction visualizations...")
    labels = np.concatenate([
        np.zeros(len(fake_features)),
        np.ones(len(real_features))
    ])
    
    # PCA visualization
    create_dimensionality_reduction_plot(
        all_features, labels, save_path, method='pca'
    )
    
    # t-SNE visualization
    create_dimensionality_reduction_plot(
        all_features, labels, save_path, method='tsne'
    )
    
    return results

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Generate sample data
    fake_data = np.random.normal(0, 1, (n_samples, n_features))
    real_data = np.random.normal(0.5, 1, (n_samples, n_features))
    feature_names = [f"Feature_{i+1}" for i in range(n_features)]
    
    # Run analysis
    results = analyze_features(
        fake_data,
        real_data,
        feature_names,
        save_folder="test_plots"
    )
    
    # Print results
    for feature, stats in results.items():
        print(f"\nResults for {feature}:")
        for test, pval in stats.items():
            print(f"  {test}: {pval:.4e}")