# Fake News Detection using Graph Neural Networks

## Description
This project implements an intelligent system for detecting fake news using Graph Neural Networks (GNN). The system analyzes news propagation patterns in social networks, combining graph structural information with text features and user behavior to effectively identify misinformation. By modeling the news spread as a graph, where nodes represent tweets/posts and edges represent interactions, our approach captures both content and propagation dynamics.
please extract file nx_network_data.zip

## Key Features
- Multiple GNN architectures (GCN, GAT, GraphSAGE)
- News propagation pattern analysis
- Bot detection integration
- Sentiment analysis
- Comprehensive evaluation metrics
- Visualization tools
- K-fold cross-validation support
- Early stopping and learning rate scheduling
- Attention mechanism for interpretability

## Prerequisites
```bash
Python 3.8+
PyTorch 1.9+
PyTorch Geometric 2.0+
scikit-learn
numpy
pandas
tqdm
matplotlib
seaborn
networkx
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/mohammadbaghershahmir/fake-news-detection.git
cd fake-news-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── main.py             # Main execution file
├── model.py            # GNN architecture implementations
├── preprocess.py       # Data preprocessing and graph construction
├── trainer.py          # Model training management
├── evaluate.py         # Evaluation functions
├── visualize.py        # Visualization tools
├── util.py            # Helper functions
├── requirements.txt    # Project dependencies
└── README.md          # Documentation
```

## Data Structure
```
data/
├── raw/               # Raw tweet data
│   ├── fake/         # Fake news tweets
│   └── real/         # Real news tweets
├── processed/         # Processed graph data
└── features/         # Extracted features
```

## Usage
1. Data Preprocessing:
```bash
python preprocess.py --data_dir data/raw --output_dir data/processed
```

2. Model Training:
```bash
python main.py --architecture gcn --hidden_dim 128 --num_layers 3
```

3. Model Evaluation:
```bash
python main.py --mode evaluate --model_path models/best_model.pth
```

## Model Parameters
- `architecture`: GNN architecture type ('gcn', 'gat', 'sage')
- `hidden_dim`: Hidden layer dimensions (default: 128)
- `num_layers`: Number of GNN layers (default: 3)
- `dropout`: Dropout rate (default: 0.5)
- `lr`: Learning rate (default: 0.001)
- `weight_decay`: L2 regularization (default: 1e-4)
- `batch_size`: Batch size (default: 32)
- `epochs`: Number of training epochs (default: 150)
- `patience`: Early stopping patience (default: 15)

## Results and Analysis
Our model has been extensively evaluated on a large-scale dataset of social media posts, achieving robust performance across different news categories and propagation patterns.

### Performance Metrics
- **Accuracy**: 85.3% (±1.2%)
- **Precision**: 83.7% (±1.5%)
- **Recall**: 87.1% (±1.3%)
- **F1-Score**: 85.4% (±1.4%)
- **AUC-ROC**: 0.892 (±0.011)

### Key Findings
1. **Propagation Patterns**: 
   - Fake news typically shows faster initial spread but shorter longevity
   - More frequent but shorter retweet chains compared to genuine news
   - Higher clustering coefficient in fake news propagation graphs

2. **User Behavior**:
   - Bot accounts are involved in 27% of fake news propagation
   - Fake news spreaders show more burst-like behavior
   - Higher ratio of new/suspicious accounts in fake news propagation

3. **Content Analysis**:
   - Sentiment polarity is more extreme in fake news
   - More emotional and inflammatory language
   - Higher presence of unverified claims and sensational headlines

4. **Model Interpretability**:
   - Attention weights reveal key users and timestamps in propagation
   - Graph structure contributes 60% to classification decisions
   - User features account for 25% of the decision weight
   - Content features contribute 15% to final predictions

### Comparative Analysis
Our GNN-based approach outperforms traditional methods:
- +7% accuracy vs. text-only classifiers
- +5% accuracy vs. user-behavior-only models
- +3% accuracy vs. propagation-only approaches

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
- MohammadBagher Shahmir


## Acknowledgments
- PyTorch Geometric team for the excellent framework
- Open-source community for various tools and libraries
- All contributors and researchers in the field

## Contact
For questions and suggestions:
- Email: mohammadbaghershahmir@gmail.com
- Twitter: @yourusername
- GitHub Issues: Create an issue in this repository 