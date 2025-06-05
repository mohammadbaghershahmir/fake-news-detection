import os
import logging
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
import sys

# تنظیم logging برای نمایش پیشرفت کار
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fake_news_detection.log')
    ]
)
logger = logging.getLogger(__name__)

def check_cuda():
    """بررسی در دسترس بودن CUDA"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device

def check_dataset_path(dataset_dir: str = "data/nx_network_data"):
    """Check if dataset directory exists"""
    path = Path(dataset_dir)
    if not path.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return False
    return True

def main():
    """Main program function"""
    try:
        logger.info("Starting fake news detection pipeline...")
        
        # Check CUDA
        device = check_cuda()
        
        # Check data paths
        dataset_dir = "data/nx_network_data"
        sample_ids_dir = "data/sample_ids"
        
        if not check_dataset_path(dataset_dir):
            logger.error("Please make sure the dataset directory exists")
            return
            
        if not check_dataset_path(sample_ids_dir):
            logger.error("Please make sure the sample IDs directory exists")
            return
        
        # Model settings
        model_params = {
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.5,
            'architecture': 'gcn',
            'n_splits': 5,
            'batch_size': 32,
            'epochs': 150,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'patience': 15
        }
        
        logger.info("Model parameters:")
        for param, value in model_params.items():
            logger.info(f"  {param}: {value}")
        
        # Check directory structure
        logger.info("Checking directory structure...")
        required_dirs = ['data', 'experiments', 'models', 'plots']
        for dir_name in required_dirs:
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Directory '{dir_name}' is ready")
        
        # Display system information
        logger.info("\nSystem Information:")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"NumPy version: {np.__version__}")
        
        # Main pipeline execution
        logger.info("\nStarting main pipeline...")
        
        # 1. Load data
        logger.info("1. Loading dataset...")
        from load_dataset import load_dataset
        fake_samples, real_samples = load_dataset(dataset_dir, "politifact", sample_ids_dir)
        logger.info(f"Loaded {len(fake_samples)} fake and {len(real_samples)} real samples")
        
        # 2. Preprocess data
        logger.info("\n2. Preprocessing data...")
        from preprocess import preprocess_samples
        train_data, val_data, test_data = preprocess_samples(fake_samples, real_samples, 
                                                           model_params['n_splits'])
        logger.info("Data preprocessing completed")
        
        # 3. Build and train model
        logger.info("\n3. Training model...")
        from model import FakeNewsDetector
        from trainer import ModelTrainer
        
        model = FakeNewsDetector(
            input_dim=train_data.num_features,
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            dropout=model_params['dropout'],
            architecture=model_params['architecture']
        ).to(device)
        
        trainer = ModelTrainer(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=device,
            **model_params
        )
        
        best_model = trainer.train()
        logger.info("Model training completed")
        
        # 4. Evaluate model
        logger.info("\n4. Evaluating model...")
        from evaluate import evaluate_model
        metrics = evaluate_model(best_model, test_data, device)
        
        logger.info("Test Set Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
            
        # 5. Analyze and visualize results
        logger.info("\n5. Analyzing and visualizing results...")
        from visualize import create_visualization_report
        
        save_dir = "plots"
        create_visualization_report(
            history=trainer.history,
            metrics=metrics,
            model=best_model,
            feature_names=train_data.feature_names,
            fake_samples=fake_samples,
            real_samples=real_samples,
            save_dir=save_dir
        )
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())