"""
Constants for the Fake News Detection project.
Defines node and edge types for propagation graphs.
"""

# Edge types for propagation graph
RETWEET_EDGE: str = "retweet"
REPLY_EDGE: str = "reply"

# Node types for propagation graph
NEWS_ROOT_NODE: int = 1
POST_NODE: int = 2
RETWEET_NODE: int = 3
REPLY_NODE: int = 4