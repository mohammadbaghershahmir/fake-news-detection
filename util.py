from datetime import datetime
import time
from typing import Optional, List, Set, Dict, Union
from constants import RETWEET_NODE, REPLY_NODE, NEWS_ROOT_NODE, POST_NODE
from dataclasses import dataclass
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TweetNode:
    """کلاس نگهداری اطلاعات یک توییت در درخت انتشار"""
    
    tweet_id: str
    text: str
    user_id: str
    timestamp: str
    node_type: str  # 'root', 'retweet', یا 'reply'
    sentiment: Optional[Dict[str, float]] = None  # {'positive': float, 'negative': float, 'neutral': float}
    botometer_score: Optional[float] = None
    retweet_children: List['TweetNode'] = None
    reply_children: List['TweetNode'] = None
    
    def __post_init__(self):
        """مقداردهی اولیه لیست‌های فرزندان"""
        if self.retweet_children is None:
            self.retweet_children = []
        if self.reply_children is None:
            self.reply_children = []

def load_json_data(file_path: Union[str, Path]) -> Dict:
    """بارگذاری داده‌ها از فایل JSON
    
    Args:
        file_path: مسیر فایل JSON
        
    Returns:
        Dict: داده‌های بارگذاری شده
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        raise

def build_tweet_tree(data: Dict) -> TweetNode:
    """ساخت درخت انتشار از داده‌های JSON
    
    Args:
        data: داده‌های JSON شامل اطلاعات توییت و انتشار آن
        
    Returns:
        TweetNode: ریشه درخت انتشار
    """
    def create_node(tweet_data: Dict, node_type: str) -> TweetNode:
        """ساخت یک گره توییت از داده‌های JSON"""
        return TweetNode(
            tweet_id=tweet_data.get('id_str', ''),
            text=tweet_data.get('text', ''),  # استفاده از رشته خالی برای فیلدهای ناموجود
            user_id=tweet_data.get('user', {}).get('id_str', ''),
            timestamp=tweet_data.get('created_at', ''),
            node_type=node_type,
            sentiment=tweet_data.get('sentiment'),
            botometer_score=tweet_data.get('botometer_score')
        )
    
    try:
        # ساخت گره ریشه
        root = create_node(data['tweet'], 'root')
        
        # اضافه کردن retweet‌ها
        for retweet in data.get('retweets', []):
            retweet_node = create_node(retweet, 'retweet')
            root.retweet_children.append(retweet_node)
        
        # اضافه کردن reply‌ها
        for reply in data.get('replies', []):
            reply_node = create_node(reply, 'reply')
            root.reply_children.append(reply_node)
            
            # اضافه کردن retweet‌های reply‌ها
            for retweet in reply.get('retweets', []):
                retweet_node = create_node(retweet, 'retweet')
                reply_node.retweet_children.append(retweet_node)
        
        return root
        
    except Exception as e:
        logger.error(f"Error building tweet tree: {str(e)}")
        raise

def save_json_data(data: Dict, file_path: Union[str, Path]) -> None:
    """ذخیره داده‌ها در فایل JSON
    
    Args:
        data: داده‌های مورد نظر برای ذخیره
        file_path: مسیر فایل JSON
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        raise

def get_tree_statistics(root: TweetNode) -> Dict[str, int]:
    """محاسبه آمار درخت انتشار
    
    Args:
        root: ریشه درخت انتشار
        
    Returns:
        Dict[str, int]: آمار مختلف درخت
    """
    stats = {
        'total_nodes': 0,
        'max_depth': 0,
        'num_retweets': 0,
        'num_replies': 0,
        'num_bot_accounts': 0  # تعداد حساب‌های بات (botometer_score > 0.7)
    }
    
    def dfs(node: TweetNode, depth: int = 0):
        """پیمایش عمقی درخت و محاسبه آمار"""
        stats['total_nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        if node.botometer_score and node.botometer_score > 0.7:
            stats['num_bot_accounts'] += 1
            
        for child in node.retweet_children:
            stats['num_retweets'] += 1
            dfs(child, depth + 1)
            
        for child in node.reply_children:
            stats['num_replies'] += 1
            dfs(child, depth + 1)
    
    dfs(root)
    return stats

class TweetNode:
    """A class representing a node in the tweet propagation graph.

    Attributes:
        tweet_id: Unique identifier for the tweet.
        text: Content of the tweet (optional).
        created_time: Unix timestamp of tweet creation (optional).
        user_name: Username of the tweet author (optional).
        user_id: User ID of the tweet author (optional).
        news_id: ID of the associated news article (optional).
        node_type: Type of node (NEWS_ROOT_NODE, POST_NODE, RETWEET_NODE, REPLY_NODE).
        botometer_score: Bot likelihood score (optional).
        sentiment: Sentiment score of the tweet (optional).
        retweet_children: List of retweet child nodes.
        reply_children: List of reply child nodes.
        children: Set of all child nodes (retweets and replies).
        parent_node: Parent node in the graph (optional).
    """

    def __init__(
        self,
        tweet_id: str,
        text: Optional[str] = None,
        created_time: Optional[float] = None,
        user_name: Optional[str] = None,
        user_id: Optional[str] = None,
        news_id: Optional[str] = None,
        node_type: Optional[int] = None,
        botometer_score: Optional[float] = None,
        sentiment: Optional[float] = None,
    ):
        self.tweet_id = tweet_id
        self.text = text
        self.created_time = created_time
        self.user_name = user_name
        self.user_id = user_id
        self.news_id = news_id
        self.node_type = node_type
        self.botometer_score = botometer_score
        self.sentiment = sentiment
        self.retweet_children: List['TweetNode'] = []
        self.reply_children: List['TweetNode'] = []
        self.children: Set['TweetNode'] = set()
        self.parent_node: Optional['TweetNode'] = None

    def __eq__(self, other: 'TweetNode') -> bool:
        """Check equality based on tweet_id."""
        if not isinstance(other, TweetNode):
            return False
        return self.tweet_id == other.tweet_id

    def __hash__(self) -> int:
        """Hash based on tweet_id."""
        return hash(self.tweet_id)

    def set_node_type(self, node_type: int) -> None:
        """Set the node type."""
        if node_type not in {NEWS_ROOT_NODE, POST_NODE, RETWEET_NODE, REPLY_NODE}:
            raise ValueError(f"Invalid node type: {node_type}")
        self.node_type = node_type

    def set_parent_node(self, parent_node: 'TweetNode') -> None:
        """Set the parent node."""
        self.parent_node = parent_node

    def add_retweet_child(self, child_node: 'TweetNode') -> None:
        """Add a retweet child to the node."""
        self.retweet_children.append(child_node)
        self.children.add(child_node)
        child_node.set_parent_node(self)
        child_node.set_node_type(RETWEET_NODE)

    def add_reply_child(self, child_node: 'TweetNode') -> None:
        """Add a reply child to the node."""
        self.reply_children.append(child_node)
        self.children.add(child_node)
        child_node.set_parent_node(self)
        child_node.set_node_type(REPLY_NODE)

    def get_contents(self) -> dict:
        """Return a dictionary of node contents."""
        return {
            "tweet_id": str(self.tweet_id),
            "text": self.text,
            "created_time": self.created_time,
            "user_name": self.user_name,
            "user_id": self.user_id,
            "news_id": self.news_id,
        }

    def twitter_datetime_str_to_object(date_str: str) -> int:
        """Convert Twitter datetime string to Unix timestamp.
    
        Args:
            date_str: Twitter datetime string (e.g., 'Wed Jun 01 12:00:00 +0000 2023').
    
        Returns:
            Unix timestamp as an integer.
        """
        try:
            time_struct = time.strptime(date_str, "%a %b %d %H:%M:%S +0000 %Y")
            dt = datetime.fromtimestamp(time.mktime(time_struct))
            return int(dt.timestamp())
        except ValueError as e:
            raise ValueError(f"Invalid datetime format: {date_str}") from e