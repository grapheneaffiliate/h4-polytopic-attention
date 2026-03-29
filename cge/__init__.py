"""Compression-Guided Exploration — Explore, Compress, Synthesize."""
from .core import GraphExplorer, NodeInfo
from .compression import CompressionLayer
from .compression_v2 import CompressionLayerV2
from .agent import CGEAgent, BFSAgent
from .agent_v2 import CGEAgentV2
from .agent_v3 import CGEAgentV3
