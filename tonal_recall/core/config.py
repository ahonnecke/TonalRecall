"""Configuration management for Tonal Recall components."""

from typing import Dict, Any, Optional
import json
import os
from pathlib import Path

from ..logger import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Configuration manager for Tonal Recall components."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files, or None to use default
        """
        if config_dir is None:
            # Use ~/.config/tonal_recall by default
            home = os.path.expanduser("~")
            config_dir = os.path.join(home, ".config", "tonal_recall")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configurations
        self.default_configs = {
            "note_detector": {
                "min_frequency": 30.0,
                "min_confidence": 0.5,
                "min_signal": 0.001,
                "tolerance": 0.8,
                "min_stable_count": 3,
                "stability_majority": 0.6,
                "harmonic_correction": True,
            },
            "audio_input": {
                "sample_rate": 44100,
                "frames_per_buffer": 1024,
                "channels": 1,
            },
        }
        
        # Load existing configurations or create default ones
        self.configs = {}
        for config_name, default_config in self.default_configs.items():
            self.configs[config_name] = self.load_config(config_name, default_config)
    
    def load_config(self, name: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from file or create default.
        
        Args:
            name: Configuration name
            default_config: Default configuration to use if file doesn't exist
            
        Returns:
            Configuration dictionary
        """
        config_file = self.config_dir / f"{name}.json"
        
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
                
                # Ensure all default keys are present
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                
                return config
            except Exception as e:
                logger.error(f"Error loading configuration from {config_file}: {e}")
                return default_config.copy()
        else:
            # Create default configuration
            config = default_config.copy()
            self.save_config(name, config)
            return config
    
    def save_config(self, name: str, config: Dict[str, Any]) -> bool:
        """Save configuration to file.
        
        Args:
            name: Configuration name
            config: Configuration dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        config_file = self.config_dir / f"{name}.json"
        
        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration to {config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {config_file}: {e}")
            return False
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get configuration by name.
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration dictionary
        """
        return self.configs.get(name, {}).copy()
    
    def update_config(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update configuration and save to file.
        
        Args:
            name: Configuration name
            updates: Dictionary of updates to apply
            
        Returns:
            True if updated and saved successfully, False otherwise
        """
        if name not in self.configs:
            logger.error(f"Unknown configuration: {name}")
            return False
        
        # Update configuration
        self.configs[name].update(updates)
        
        # Save to file
        return self.save_config(name, self.configs[name])
    
    def reset_config(self, name: str) -> bool:
        """Reset configuration to default.
        
        Args:
            name: Configuration name
            
        Returns:
            True if reset successfully, False otherwise
        """
        if name not in self.default_configs:
            logger.error(f"Unknown configuration: {name}")
            return False
        
        # Reset to default
        self.configs[name] = self.default_configs[name].copy()
        
        # Save to file
        return self.save_config(name, self.configs[name])
