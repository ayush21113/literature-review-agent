# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentConfig:
    # Model configurations
    synthesis_model: str = "microsoft/DialoGPT-medium"
    analysis_model: str = "facebook/bart-large-cnn"
    
    # Search configurations
    max_papers: int = 5
    min_abstract_length: int = 100
    
    # Evaluation configurations
    min_review_length: int = 500
    target_gap_count: int = 3
    
    # File paths
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    
    # API configurations (if using external APIs)
    arxiv_max_results: int = 10
    request_timeout: int = 30

class ConfigManager:
    def __init__(self):
        self.config = AgentConfig()
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def get_config(self) -> AgentConfig:
        return self.config