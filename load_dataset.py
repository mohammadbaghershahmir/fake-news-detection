import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
import networkx as nx
from networkx.readwrite import json_graph
from tqdm import tqdm
import pandas as pd
from util import TweetNode
from constants import NEWS_ROOT_NODE, POST_NODE, RETWEET_NODE, REPLY_NODE, RETWEET_EDGE, REPLY_EDGE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """Class for loading and preprocessing the FakeNewsNet dataset."""

    def __init__(self, dataset_dir: str, news_source: str, sample_ids_dir: Optional[str] = None):
        """Initialize the dataset loader.

        Args:
            dataset_dir: Directory containing the dataset.
            news_source: Source of news articles (e.g., 'politifact', 'gossipcop').
            sample_ids_dir: Directory containing sample ID files. If None, defaults to dataset_dir/sample_ids.
        """
        self.dataset_dir = Path(dataset_dir)
        self.news_source = news_source
        self.sample_ids_dir = Path(sample_ids_dir) if sample_ids_dir else Path("data/sample_ids")
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input parameters and dataset structure."""
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {self.dataset_dir}")
        
        if not self.sample_ids_dir.exists():
            raise ValueError(f"Sample IDs directory does not exist: {self.sample_ids_dir}")
        
        valid_sources = {'politifact', 'gossipcop'}
        if self.news_source not in valid_sources:
            raise ValueError(f"Invalid news source. Must be one of: {valid_sources}")

    def _load_sample_ids(self, label: str) -> List[str]:
        """Load sample IDs from file with error handling.

        Args:
            label: Label of news ('fake' or 'real').

        Returns:
            List of sample IDs.
        """
        file_path = self.sample_ids_dir / f"{self.news_source}_{label}_ids_list.txt"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Sample ID file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading sample IDs: {e}")
            raise

    def _construct_tweet_node(self, json_data: Dict[str, Any]) -> TweetNode:
        """Construct a tweet node from JSON data with validation.

        Args:
            json_data: JSON data representing the tweet.

        Returns:
            TweetNode object.
        """
        required_fields = {'tweet_id', 'text', 'time', 'user'}
        missing_fields = required_fields - set(json_data.keys())
        if missing_fields:
            logger.warning(f"Missing fields in tweet data: {missing_fields}")
            # Fill missing fields with None
            for field in missing_fields:
                json_data[field] = None

        return TweetNode(
            tweet_id=json_data['tweet_id'],
            text=json_data.get('text'),
            created_time=json_data.get('time'),
            user_id=json_data.get('user'),
            node_type=json_data.get('type', POST_NODE),
            botometer_score=json_data.get('bot_score'),
            sentiment=json_data.get('sentiment')
        )

    def _build_propagation_graph(self, json_data: Dict[str, Any]) -> TweetNode:
        """Build propagation graph from JSON data.

        Args:
            json_data: JSON data representing the tweet tree.

        Returns:
            Root TweetNode object.
        """
        graph = json_graph.tree_graph(json_data)
        root_node_id = next(node for node, in_degree in graph.in_degree() if in_degree == 0)
        node_id_obj_dict: Dict[str, TweetNode] = {}
        
        def dfs_build_graph(node_id: str, visited: Set[str]) -> None:
            if node_id in visited:
                return

            visited.add(node_id)
            node_data = graph.nodes[node_id]
            tweet_node = self._construct_tweet_node(node_data)
            node_id_obj_dict[node_id] = tweet_node

            for neighbor_id in graph.successors(node_id):
                if neighbor_id not in visited:
                    dfs_build_graph(neighbor_id, visited)
                    child_node = node_id_obj_dict[neighbor_id]
                    if child_node.node_type == RETWEET_NODE:
                        tweet_node.add_retweet_child(child_node)
                    elif child_node.node_type == REPLY_NODE:
                        tweet_node.add_reply_child(child_node)

        dfs_build_graph(root_node_id, set())
        return node_id_obj_dict[root_node_id]

    def load_samples(self, label: str) -> List[TweetNode]:
        """Load samples for a given label.

        Args:
            label: Label of news ('fake' or 'real').

        Returns:
            List of TweetNode objects.
        """
        if label not in {'fake', 'real'}:
            raise ValueError("Label must be either 'fake' or 'real'")

        sample_ids = self._load_sample_ids(label)
        samples = []
        dataset_path = self.dataset_dir / f"{self.news_source}_{label}"

        for sample_id in tqdm(sample_ids, desc=f"Loading {label} samples"):
            file_path = dataset_path / f"{sample_id}.json"
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    samples.append(self._build_propagation_graph(json_data))
            except FileNotFoundError:
                logger.warning(f"Sample file not found: {file_path}")
                continue
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in file: {file_path}")
                continue
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

        return samples

    def load_dataset(self) -> Tuple[List[TweetNode], List[TweetNode]]:
        """Load both fake and real news samples.

        Returns:
            Tuple of (fake news samples, real news samples).
        """
        logger.info(f"Loading dataset from {self.dataset_dir}")
        fake_samples = self.load_samples("fake")
        real_samples = self.load_samples("real")
        
        logger.info(f"Loaded {len(fake_samples)} fake and {len(real_samples)} real samples")
        
        if not fake_samples or not real_samples:
            raise ValueError("No samples loaded for one or both classes")
            
        return fake_samples, real_samples

def load_dataset(dataset_dir: str, news_source: str, sample_ids_dir: Optional[str] = None) -> Tuple[List[TweetNode], List[TweetNode]]:
    """Convenience function to load the dataset.

    Args:
        dataset_dir: Directory containing the dataset.
        news_source: Source of news articles (e.g., 'politifact', 'gossipcop').
        sample_ids_dir: Directory containing sample ID files. If None, defaults to dataset_dir/sample_ids.

    Returns:
        Tuple of (fake news samples, real news samples).
    """
    loader = DatasetLoader(dataset_dir, news_source, sample_ids_dir)
    return loader.load_dataset()

if __name__ == "__main__":
    # Example usage and testing
    try:
        fake_samples, real_samples = load_dataset("data/nx_network_data", "politifact")
        print(f"Successfully loaded {len(fake_samples)} fake and {len(real_samples)} real samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")