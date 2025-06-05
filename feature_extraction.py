import queue
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.spatial.distance import cosine
from util import TweetNode
from constants import RETWEET_EDGE, REPLY_EDGE, NEWS_ROOT_NODE, POST_NODE, RETWEET_NODE, REPLY_NODE
import torch
from torch_geometric.data import Data

class BaseFeatureHelper:
    """Base class for feature extraction helpers."""
    
    def get_feature_group_name(self) -> str:
        raise NotImplementedError

    def get_micro_feature_method_references(self) -> List[callable]:
        raise NotImplementedError

    def get_micro_feature_method_names(self) -> List[str]:
        raise NotImplementedError

    def get_micro_feature_short_names(self) -> List[str]:
        raise NotImplementedError

    def get_macro_feature_method_references(self) -> List[callable]:
        raise NotImplementedError

    def get_macro_feature_method_names(self) -> List[str]:
        raise NotImplementedError

    def get_macro_feature_short_names(self) -> List[str]:
        raise NotImplementedError

    def get_dump_file_name(self, news_source: str, micro_features: bool, macro_features: bool, 
                         label: str, file_dir: str) -> str:
        """Generate filename for caching features."""
        file_tags = [news_source, label, self.get_feature_group_name()]
        if micro_features:
            file_tags.append("micro")
        if macro_features:
            file_tags.append("macro")
        return f"{file_dir}/{ '_'.join(file_tags) }.pkl"

    def get_features_array(self, prop_graphs: List[TweetNode], micro_features: bool, macro_features: bool,
                         news_source: Optional[str] = None, label: Optional[str] = None,
                         file_dir: str = "data/features", use_cache: bool = False) -> np.ndarray:
        """Extract features and return as a NumPy array."""
        file_name = self.get_dump_file_name(news_source, micro_features, macro_features, label, file_dir)
        data_file = Path(file_name)

        if use_cache and data_file.is_file():
            with open(file_name, "rb") as f:
                return pickle.load(f)

        function_refs = []
        if micro_features:
            function_refs.extend(self.get_micro_feature_method_references())
        if macro_features:
            function_refs.extend(self.get_macro_feature_method_references())

        if not function_refs:
            return np.array([])

        all_features = []

        for func in function_refs:
            features = []
            for graph in prop_graphs:
                value = func(graph)
                try:
                    features.append(float(np.ravel(value)[0]))
                except Exception as e:
                    print(f"⚠️ Error converting value to float: {value} (type: {type(value)}) - {e}")
                    features.append(0.0)  # یا مقدار پیش‌فرض
            all_features.append(features)
        feature_array = np.transpose(np.array(all_features))
        with open(file_name, "wb") as f:
            pickle.dump(feature_array, f)

        return feature_array

class StructureFeatureHelper(BaseFeatureHelper):
    """Helper class for extracting structural features from propagation graphs."""

    def get_feature_group_name(self) -> str:
        return "struct"

    def get_micro_feature_method_references(self) -> List[callable]:
        return [
            self.get_tree_height,
            self.get_node_count,
            self.get_max_outdegree,
            self.get_num_cascades_with_replies,
            self.get_fraction_cascades_with_replies
        ]

    def get_micro_feature_method_names(self) -> List[str]:
        return [
            "Micro - Tree depth (replies)",
            "Micro - Number of nodes (replies)",
            "Micro - Maximum out-degree (replies)",
            "Number of cascades with replies",
            "Fraction of cascades with replies"
        ]

    def get_micro_feature_short_names(self) -> List[str]:
        return ["S10", "S11", "S12", "S13", "S14"]

    def get_macro_feature_method_references(self) -> List[callable]:
        return [
            self.get_tree_height,
            self.get_node_count,
            self.get_max_outdegree,
            self.get_num_cascades,
            self.get_max_outdegree_depth,
            self.get_num_cascades_with_retweets,
            self.get_fraction_cascades_with_retweets,
            self.get_num_bot_users,
            self.get_fraction_bot_users
        ]

    def get_macro_feature_method_names(self) -> List[str]:
        return [
            "Macro - Tree depth (retweets)",
            "Macro - Number of nodes (retweets)",
            "Macro - Maximum out-degree (retweets)",
            "Number of cascades (retweets)",
            "Depth of max out-degree node",
            "Number of cascades with retweets",
            "Fraction of cascades with retweets",
            "Number of bot users retweeting",
            "Fraction of bot users retweeting"
        ]

    def get_macro_feature_short_names(self) -> List[str]:
        return ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]

    def get_tree_height(self, node: TweetNode, edge_type: str = RETWEET_EDGE) -> int:
        """Calculate the height of the tree for a given edge type."""
        if not node:
            return 0
        children = node.retweet_children if edge_type == RETWEET_EDGE else node.reply_children
        return max([StructureFeatureHelper().get_tree_height(child, edge_type) for child in children], default=0) + 1

    def get_node_count(self, node: TweetNode, edge_type: str = RETWEET_EDGE) -> int:
        """Count the number of nodes in the tree for a given edge type."""
        if not node:
            return 0
        children = node.retweet_children if edge_type == RETWEET_EDGE else node.reply_children
        return sum(self.get_node_count(child, edge_type) for child in children) + 1

    def get_max_outdegree(self, node: TweetNode, edge_type: str = RETWEET_EDGE) -> int:
        """Get the maximum out-degree in the tree."""
        if not node:
            return 0
        children = node.retweet_children if edge_type == RETWEET_EDGE else node.reply_children
        max_degree = len(children) if node.node_type != NEWS_ROOT_NODE else 0
        for child in children:
            max_degree = max(max_degree, self.get_max_outdegree(child, edge_type))
        return max_degree

    def get_num_cascades(self, node: TweetNode) -> int:
        """Get the number of cascades (retweet children)."""
        return len(node.retweet_children)

    def get_max_outdegree_depth(self, node: TweetNode) -> Tuple[Optional[TweetNode], int]:
        """Get the node and its out-degree with maximum out-degree."""
        def find_max_outdegree_node(n: TweetNode) -> Tuple[Optional[TweetNode], int]:
            if not n:
                return None, 0
            children = n.retweet_children
            max_node, max_degree = (n if n.node_type != NEWS_ROOT_NODE else None), len(children)
            for child in children:
                child_node, child_degree = find_max_outdegree_node(child)
                if child_degree > max_degree:
                    max_node, max_degree = child_node, child_degree
            return max_node, max_degree
    
        return find_max_outdegree_node(node)



        def get_depth(root: TweetNode, target: TweetNode, level: int = 0) -> int:
            if root.tweet_id == target.tweet_id:
                return level
            for child in root.retweet_children:
                result = get_depth(child, target, level + 1)
                if result != 0:
                    return result
            return 0

        max_node, _ = find_max_outdegree_node(node)
        return get_depth(node, max_node) if max_node else 0

    def get_num_cascades_with_retweets(self, node: TweetNode) -> int:
        """Count cascades with at least one retweet."""
        return sum(1 for child in node.retweet_children if len(child.retweet_children) > 0)

    def get_fraction_cascades_with_retweets(self, node: TweetNode) -> float:
        """Get fraction of cascades with retweets."""
        total = len(node.retweet_children)
        return self.get_num_cascades_with_retweets(node) / total if total > 0 else 0

    def get_num_cascades_with_replies(self, node: TweetNode) -> int:
        """Count cascades with at least one reply."""
        return sum(1 for child in node.reply_children if len(child.reply_children) > 0)

    def get_fraction_cascades_with_replies(self, node: TweetNode) -> float:
        """Get fraction of cascades with replies."""
        total = len(node.reply_children)
        return self.get_num_cascades_with_replies(node) / total if total > 0 else 0

    def get_num_bot_users(self, node: TweetNode) -> int:
        """Count number of bot users in retweet children."""
        q = queue.Queue()
        q.put(node)
        bot_count = 0
        while not q.empty():
            n = q.get()
            for child in n.retweet_children:
                q.put(child)
                if child.node_type == RETWEET_NODE and child.botometer_score and child.botometer_score > 0.5:
                    bot_count += 1
        return bot_count

    def get_fraction_bot_users(self, node: TweetNode) -> float:
        """Get fraction of bot users in retweet children."""
        q = queue.Queue()
        q.put(node)
        bot_count, human_count = 1, 1
        while not q.empty():
            n = q.get()
            for child in n.retweet_children:
                q.put(child)
                if child.node_type == RETWEET_NODE and child.botometer_score:
                    if child.botometer_score > 0.5:
                        bot_count += 1
                    else:
                        human_count += 1
        return bot_count / (bot_count + human_count)

class TemporalFeatureHelper(BaseFeatureHelper):
    """Helper class for extracting temporal features from propagation graphs."""

    def get_feature_group_name(self) -> str:
        return "temp"

    def get_micro_feature_method_references(self) -> List[callable]:
        return [
            self.get_avg_time_between_replies,
            self.get_time_diff_first_post_first_reply,
            self.get_time_diff_first_post_last_reply,
            self.get_avg_time_between_replies_deepest_cascade,
            self.get_time_diff_post_last_reply_deepest_cascade
        ]

    def get_micro_feature_method_names(self) -> List[str]:
        return [
            "Average time between adjacent replies",
            "Time diff between first post and first reply",
            "Time diff between first post and last reply",
            "Average time between replies in deepest cascade",
            "Time diff between first post and last reply in deepest cascade"
        ]

    def get_micro_feature_short_names(self) -> List[str]:
        return ["T9", "T10", "T11", "T12", "T13"]

    def get_macro_feature_method_references(self) -> List[callable]:
        return [
            self.get_avg_time_between_retweets,
            self.get_time_diff_first_post_last_retweet,
            self.get_time_diff_first_post_max_outdegree,
            self.get_time_diff_first_last_post,
            self.get_time_diff_post_last_retweet_deepest_cascade,
            self.get_avg_time_retweet_deepest_cascade,
            self.get_avg_time_between_posts,
            self.get_time_diff_first_post_first_retweet
        ]

    def get_macro_feature_method_names(self) -> List[str]:
        return [
            "Average time between adjacent retweets",
            "Time diff between first post and last retweet",
            "Time diff between first post and max out-degree node",
            "Time diff between first and last post",
            "Time diff between post and last retweet in deepest cascade",
            "Average time between retweets in deepest cascade",
            "Average time between posts",
            "Time diff between first post and first retweet"
        ]

    def get_macro_feature_short_names(self) -> List[str]:
        return ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]

    def get_first_post_time(self, node: TweetNode) -> float:
        """Get the earliest post time in the graph."""
        return min([child.created_time for child in node.children], default=float('inf'))

    def get_avg_time_between_replies(self, node: TweetNode) -> float:
        """Calculate average time between adjacent replies."""
        q = queue.Queue()
        q.put(node)
        time_diffs = []
        while not q.empty():
            n = q.get()
            for child in n.reply_children:
                q.put(child)
                if n.node_type == REPLY_NODE and child.node_type == REPLY_NODE:
                    time_diffs.append(child.created_time - n.created_time)
        return np.mean(time_diffs) if time_diffs else 0

    def get_time_diff_first_post_first_reply(self, node: TweetNode) -> float:
        """Calculate time difference between first post and first reply."""
        first_post_time = self.get_first_post_time(node)
        first_reply_time = float('inf')
        q = queue.Queue()
        q.put(node)
        while not q.empty():
            n = q.get()
            for child in n.reply_children:
                q.put(child)
                if child.node_type == REPLY_NODE:
                    first_reply_time = min(first_reply_time, child.created_time)
        return first_reply_time - first_post_time if first_reply_time != float('inf') else 0

    def get_time_diff_first_post_last_reply(self, node: TweetNode) -> float:
        """Calculate time difference between first post and last reply."""
        first_post_time = self.get_first_post_time(node)
        last_reply_time = 0
        q = queue.Queue()
        q.put(node)
        while not q.empty():
            n = q.get()
            for child in n.reply_children:
                q.put(child)
                if child.node_type == REPLY_NODE:
                    last_reply_time = max(last_reply_time, child.created_time)
        return last_reply_time - first_post_time if last_reply_time != 0 else 0

    def get_avg_time_between_replies_deepest_cascade(self, node: TweetNode) -> float:
        """Calculate average time between replies in the deepest cascade."""
        max_height, max_node = 0, None
        for child in node.reply_children:
            height = StructureFeatureHelper().get_tree_height(child, REPLY_EDGE)
            if height > max_height:
                max_height, max_node = height, child
        if not max_node:
            return 0
        return self.get_avg_time_between_replies(max_node)

    def get_time_diff_post_last_reply_deepest_cascade(self, node: TweetNode) -> float:
        """Calculate time difference between post and last reply in deepest cascade."""
        max_height, max_node = 0, None
        for child in node.reply_children:
            height = StructureFeatureHelper().get_tree_height(child, REPLY_EDGE)
            if height > max_height:
                max_height, max_node = height, child
        if not max_node:
            return 0
        first_time = max_node.created_time
        last_time = 0
        q = queue.Queue()
        q.put(max_node)
        while not q.empty():
            n = q.get()
            for child in n.reply_children:
                q.put(child)
                if child.node_type == REPLY_NODE:
                    last_time = max(last_time, child.created_time)
        return last_time - first_time if last_time != 0 else 0

    def get_avg_time_between_retweets(self, node: TweetNode) -> float:
        """Calculate average time between adjacent retweets."""
        q = queue.Queue()
        q.put(node)
        time_diffs = []
        while not q.empty():
            n = q.get()
            for child in n.retweet_children:
                q.put(child)
                if n.node_type == RETWEET_NODE and child.node_type == RETWEET_NODE:
                    time_diffs.append(child.created_time - n.created_time)
        return np.mean(time_diffs) if time_diffs else 0

    def get_time_diff_first_post_last_retweet(self, node: TweetNode) -> float:
        """Calculate time difference between first post and last retweet."""
        first_post_time = self.get_first_post_time(node)
        last_retweet_time = 0
        q = queue.Queue()
        q.put(node)
        while not q.empty():
            n = q.get()
            for child in n.retweet_children:
                q.put(child)
                if child.node_type == RETWEET_NODE:
                    last_retweet_time = max(last_retweet_time, child.created_time)
        return last_retweet_time - first_post_time if last_retweet_time != 0 else 0

    def get_time_diff_first_post_max_outdegree(self, node: TweetNode) -> float:
        """Calculate time difference between first post and max out-degree node."""
        max_node, _ = StructureFeatureHelper().get_max_outdegree_depth(node)
        first_post_time = self.get_first_post_time(node)
        return max_node.created_time - first_post_time if max_node else 0

    def get_time_diff_first_last_post(self, node: TweetNode) -> float:
        """Calculate time difference between first and last post."""
        posts = sorted(node.children, key=lambda x: x.created_time or float('inf'))
        return posts[-1].created_time - posts[0].created_time if len(posts) > 1 else 0

    def get_time_diff_post_last_retweet_deepest_cascade(self, node: TweetNode) -> float:
        """Calculate time difference between post and last retweet in deepest cascade."""
        max_height, max_node = 0, None
        for child in node.retweet_children:
            height = StructureFeatureHelper().get_tree_height(child, RETWEET_EDGE)
            if height > max_height:
                max_height, max_node = height, child
        if not max_node:
            return 0
        first_time = max_node.created_time
        last_time = 0
        q = queue.Queue()
        q.put(max_node)
        while not q.empty():
            n = q.get()
            for child in n.retweet_children:
                q.put(child)
                if child.node_type == RETWEET_NODE:
                    last_time = max(last_time, child.created_time)
        return last_time - first_time if last_time != 0 else 0

    def get_avg_time_retweet_deepest_cascade(self, node: TweetNode) -> float:
        """Calculate average time between retweets in deepest cascade."""
        max_height, max_node = 0, None
        for child in node.retweet_children:
            height = StructureFeatureHelper().get_tree_height(child, RETWEET_EDGE)
            if height > max_height:
                max_height, max_node = height, child
        if not max_node:
            return 0
        return self.get_avg_time_between_retweets(max_node)

    def get_avg_time_between_posts(self, node: TweetNode) -> float:
        """Calculate average time between posts."""
        posts = sorted(node.children, key=lambda x: x.created_time or float('inf'))
        if len(posts) <= 1:
            return 0
        time_diffs = [posts[i+1].created_time - posts[i].created_time for i in range(len(posts)-1)]
        return np.mean(time_diffs)

    def get_time_diff_first_post_first_retweet(self, node: TweetNode) -> float:
        """Calculate time difference between first post and first retweet."""
        first_post_time = self.get_first_post_time(node)
        first_retweet_time = float('inf')
        q = queue.Queue()
        q.put(node)
        while not q.empty():
            n = q.get()
            for child in n.retweet_children:
                q.put(child)
                if child.node_type == RETWEET_NODE:
                    first_retweet_time = min(first_retweet_time, child.created_time)
        return first_retweet_time - first_post_time if first_retweet_time != float('inf') else 0

class LinguisticFeatureHelper(BaseFeatureHelper):
    """Helper class for extracting linguistic features from propagation graphs."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_feature_group_name(self) -> str:
        return "ling"

    def get_micro_feature_method_references(self) -> List[callable]:
        return [
            self.get_reply_sentiment_ratio,
            self.get_avg_reply_sentiment,
            self.get_avg_first_level_reply_sentiment,
            self.get_avg_deepest_cascade_reply_sentiment,
            self.get_avg_deepest_cascade_first_reply_sentiment
        ]

    def get_micro_feature_method_names(self) -> List[str]:
        return [
            "Sentiment ratio of all replies",
            "Average sentiment of all replies",
            "Average sentiment of first-level replies",
            "Average sentiment of replies in deepest cascade",
            "Average sentiment of first-level replies in deepest cascade"
        ]

    def get_micro_feature_short_names(self) -> List[str]:
        return ["L1", "L2", "L3", "L4", "L5"]

    def get_macro_feature_method_references(self) -> List[callable]:
        return []

    def get_macro_feature_method_names(self) -> List[str]:
        return []

    def get_macro_feature_short_names(self) -> List[str]:
        return []

    def get_reply_sentiment_ratio(self, node: TweetNode) -> float:
        """Calculate the ratio of positive to negative reply sentiments."""
        q = queue.Queue()
        q.put(node)
        positive, negative = 1, 1
        while not q.empty():
            n = q.get()
            for child in n.reply_children:
                q.put(child)
                if child.node_type == REPLY_NODE and child.sentiment:
                    if child.sentiment > 0.05:
                        positive += 1
                    elif child.sentiment < -0.05:
                        negative += 1
        return positive / negative

    def get_avg_reply_sentiment(self, node: TweetNode) -> float:
        """Calculate average sentiment of all replies."""
        q = queue.Queue()
        q.put(node)
        sentiments = []
        while not q.empty():
            n = q.get()
            for child in n.reply_children:
                q.put(child)
                if child.node_type == REPLY_NODE and child.sentiment:
                    sentiments.append(child.sentiment)
        return np.mean(sentiments) if sentiments else 0

    def get_avg_first_level_reply_sentiment(self, node: TweetNode) -> float:
        """Calculate average sentiment of first-level replies."""
        sentiments = [
            child.sentiment for child in node.reply_children 
            if child.node_type == REPLY_NODE and child.sentiment
        ]
        return np.mean(sentiments) if sentiments else 0

    def get_avg_deepest_cascade_reply_sentiment(self, node: TweetNode) -> float:
        """Calculate average sentiment of replies in the deepest cascade."""
        max_height, max_node = 0, None
        for child in node.reply_children:
            height = StructureFeatureHelper().get_tree_height(child, REPLY_EDGE)
            if height > max_height:
                max_height, max_node = height, child
        if not max_node:
            return 0
        return self.get_avg_reply_sentiment(max_node)

    def get_avg_deepest_cascade_first_reply_sentiment(self, node: TweetNode) -> float:
        """Calculate average sentiment of first-level replies in deepest cascade."""
        max_height, max_node = 0, None
        for child in node.reply_children:
            height = StructureFeatureHelper().get_tree_height(child, REPLY_EDGE)
            if height > max_height:
                max_height, max_node = height, child
        if not max_node:
            return 0
        return self.get_avg_first_level_reply_sentiment(max_node)

def extract_all_features(
    graphs: List[TweetNode],
    news_source: str,
    label: str,
    micro_features: bool = True,
    macro_features: bool = True,
    file_dir: str = "data/features",
    use_cache: bool = True
) -> np.ndarray:
    """Extract all structural, temporal, and linguistic features."""
    struct_helper = StructureFeatureHelper()
    temp_helper = TemporalFeatureHelper()
    ling_helper = LinguisticFeatureHelper()

    struct_features = struct_helper.get_features_array(
        graphs, micro_features, macro_features, news_source, label, file_dir, use_cache
    )
    temp_features = temp_helper.get_features_array(
        graphs, micro_features, macro_features, news_source, label, file_dir, use_cache
    )
    ling_features = ling_helper.get_features_array(
        graphs, micro_features, macro_features, news_source, label, file_dir, use_cache
    )

    return np.concatenate([struct_features, temp_features, ling_features], axis=1)

def create_pyg_data(
    graphs: List[TweetNode],
    features: np.ndarray,
    labels: np.ndarray
) -> List[Data]:
    """Convert graphs and features to PyTorch Geometric Data objects.

    Args:
        graphs: List of TweetNode objects representing propagation graphs.
        features: Array of node features.
        labels: Array of labels (0 for fake, 1 for real).

    Returns:
        List of PyTorch Geometric Data objects.
    """
    data_list = []
    for i, (graph, feature, label) in enumerate(zip(graphs, features, labels)):
        # Initialize node-to-index mapping
        node_to_index = {}
        current_index = 0
        nodes = []

        # Collect all valid nodes using BFS
        q = queue.Queue()
        if not graph or not graph.tweet_id:
            print(f"Skipping graph {i}: Invalid root node or null tweet_id")
            continue
        q.put(graph)
        seen = set([graph.tweet_id])
        while not q.empty():
            node = q.get()
            # Validate node
            if not node.tweet_id:
                print(f"Warning: Skipping node with null tweet_id in graph {i}")
                continue
            # Assign index to node
            if node.tweet_id not in node_to_index:
                node_to_index[node.tweet_id] = current_index
                nodes.append(node)
                current_index += 1
            # Add valid children to queue
            for child in node.retweet_children + node.reply_children:
                if child and child.tweet_id and child.tweet_id not in seen:
                    seen.add(child.tweet_id)
                    q.put(child)
                elif not child.tweet_id:
                    print(f"Warning: Child with null tweet_id in graph {i}, skipping")

        # Build edge_index with strict validation
        edge_index = []
        valid_nodes = set(node_to_index.keys())
        for node in nodes:
            for child in node.retweet_children + node.reply_children:
                if not child or not child.tweet_id:
                    print(f"Warning: Skipping invalid child in graph {i}")
                    continue
                if child.tweet_id in valid_nodes:
                    src_idx = node_to_index[node.tweet_id]
                    dst_idx = node_to_index[child.tweet_id]
                    if src_idx < len(nodes) and dst_idx < len(nodes):
                        edge_index.append([src_idx, dst_idx])
                    else:
                        print(f"Warning: Invalid edge indices [{src_idx}, {dst_idx}] in graph {i}, skipping")
                else:
                    print(f"Warning: Child tweet_id {child.tweet_id} not in valid nodes for graph {i}, skipping")

        # Skip invalid graphs
        if not nodes:
            print(f"Skipping graph {i}: No valid nodes found")
            continue
        if not edge_index:
            print(f"Skipping graph {i}: No valid edges found")
            continue

        # Convert to tensor and validate
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        num_nodes = len(nodes)
        if edge_index.numel() == 0:
            print(f"Skipping graph {i}: Empty edge_index after validation")
            continue
        max_index = edge_index.max().item()
        if max_index >= num_nodes:
            print(f"Error in graph {i}: max edge index {max_index} exceeds num_nodes {num_nodes}")
            print(f"edge_index: {edge_index.tolist()}")
            continue

        # Create Data object
        x = torch.tensor(feature, dtype=torch.float).reshape(1, -1)
        y = torch.tensor([label], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
        try:
            data.validate(raise_on_error=True)
        except ValueError as e:
            print(f"Error in graph {i}: Validation failed - {str(e)}")
            continue
        data_list.append(data)
        print(f"Graph {i}: {num_nodes} nodes, {edge_index.shape[1]} edges")

    if not data_list:
        raise ValueError("No valid graphs were created. Check input data.")
    return data_list