"""Factory for creating Tonal Recall components."""

from typing import Optional, Dict, Type

from ..logger import get_logger
from ..audio.note_detector import NoteDetector
from ..audio.note_detection_service import NoteDetectionService
from ..audio.audio_input import SoundDeviceInput
from .config import ConfigManager
from .interfaces import INoteDetector, IAudioInput, INoteDetectionService

logger = get_logger(__name__)


class ComponentFactory:
    """Factory for creating Tonal Recall components."""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize the component factory.
        
        Args:
            config_manager: Configuration manager, or None to create a default one
        """
        self.config_manager = config_manager or ConfigManager()
        
        # Register default component implementations
        self.note_detector_classes: Dict[str, Type[INoteDetector]] = {
            "default": NoteDetector,
        }
        
        self.audio_input_classes: Dict[str, Type[IAudioInput]] = {
            "default": SoundDeviceInput,
        }
        
        self.note_detection_service_classes: Dict[str, Type[INoteDetectionService]] = {
            "default": NoteDetectionService,
        }
    
    def create_note_detector(
        self, implementation: str = "default", **kwargs
    ) -> INoteDetector:
        """Create a note detector.
        
        Args:
            implementation: Name of the implementation to use
            **kwargs: Additional parameters to pass to the constructor
            
        Returns:
            Note detector instance
            
        Raises:
            ValueError: If the implementation is not registered
        """
        if implementation not in self.note_detector_classes:
            raise ValueError(f"Unknown note detector implementation: {implementation}")
        
        # Get default configuration
        config = self.config_manager.get_config("note_detector")
        
        # Override with provided parameters
        config.update(kwargs)
        
        # Create instance
        cls = self.note_detector_classes[implementation]
        instance = cls(**config)
        
        logger.info(f"Created note detector: {implementation}")
        return instance
    
    def create_audio_input(
        self, implementation: str = "default", **kwargs
    ) -> IAudioInput:
        """Create an audio input.
        
        Args:
            implementation: Name of the implementation to use
            **kwargs: Additional parameters to pass to the constructor
            
        Returns:
            Audio input instance
            
        Raises:
            ValueError: If the implementation is not registered
        """
        if implementation not in self.audio_input_classes:
            raise ValueError(f"Unknown audio input implementation: {implementation}")
        
        # Get default configuration
        config = self.config_manager.get_config("audio_input")
        
        # Override with provided parameters
        config.update(kwargs)
        
        # Create instance
        cls = self.audio_input_classes[implementation]
        instance = cls(**config)
        
        logger.info(f"Created audio input: {implementation}")
        return instance
    
    def create_note_detection_service(
        self, implementation: str = "default", **kwargs
    ) -> INoteDetectionService:
        """Create a note detection service.
        
        Args:
            implementation: Name of the implementation to use
            **kwargs: Additional parameters to pass to the constructor
            
        Returns:
            Note detection service instance
            
        Raises:
            ValueError: If the implementation is not registered
        """
        if implementation not in self.note_detection_service_classes:
            raise ValueError(f"Unknown note detection service implementation: {implementation}")
        
        # Create audio input and note detector if not provided
        if "audio_input" not in kwargs:
            kwargs["audio_input"] = self.create_audio_input()
        
        if "note_detector" not in kwargs:
            kwargs["note_detector"] = self.create_note_detector()
        
        # Create instance
        cls = self.note_detection_service_classes[implementation]
        instance = cls(**kwargs)
        
        logger.info(f"Created note detection service: {implementation}")
        return instance
