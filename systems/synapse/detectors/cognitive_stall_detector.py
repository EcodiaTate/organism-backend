"""
EcodiaOS — Cognitive Stall Pattern Detector

Detects and tracks cognitive stalls in Synapse perception modules.
"""

from __future__ import annotations

import structlog
from typing import Dict, Any, List

from systems.evo.detectors import PatternDetector
from primitives.common import EOSBaseModel

logger = structlog.get_logger("cognitive_stall_detector")

class CognitiveStallSignature(EOSBaseModel):
    """Signature for detecting cognitive stalls."""
    perception_rate_threshold: float = 0.1
    stall_duration_cycles: int = 3

class CognitiveStallDetector(PatternDetector):
    """
    Detects cognitive stalls by monitoring perception rates
    and identifying prolonged low-activity states.
    """
    
    name = "cognitive_stall_detector"
    description = "Detect and track cognitive perception stalls"
    
    def __init__(
        self, 
        signature: CognitiveStallSignature = CognitiveStallSignature()
    ):
        """
        Initialize detector with configurable stall signature.
        
        Args:
            signature: Detection parameters for cognitive stalls
        """
        self.signature = signature
        self.stall_history: Dict[str, List[float]] = {}
    
    def scan(self, perception_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan perception data for cognitive stalls.
        
        Args:
            perception_data: Current system perception metrics
        
        Returns:
            Detected stall events
        """
        detected_stalls = {}
        
        for system, metrics in perception_data.items():
            perception_rate = metrics.get('perception_rate', 0.0)
            
            # Track perception history
            if system not in self.stall_history:
                self.stall_history[system] = []
            
            self.stall_history[system].append(perception_rate)
            
            # Limit history to last 10 cycles
            self.stall_history[system] = self.stall_history[system][-10:]
            
            # Check for stall condition
            if (
                perception_rate <= self.signature.perception_rate_threshold and
                len([r for r in self.stall_history[system] 
                     if r <= self.signature.perception_rate_threshold]) 
                >= self.signature.stall_duration_cycles
            ):
                detected_stalls[system] = {
                    "perception_rate": perception_rate,
                    "cycles_stalled": len(
                        [r for r in self.stall_history[system] 
                         if r <= self.signature.perception_rate_threshold]
                    )
                }
                
                logger.warning(
                    "Cognitive stall detected", 
                    system=system, 
                    **detected_stalls[system]
                )
        
        return detected_stalls