"""
EcodiaOS — Synapse Intent Rate Pattern Detector

Monitors and detects cognitive stalls in intent generation.
"""
from __future__ import annotations

import structlog
from typing import List, Dict, Any

from systems.evo.detectors import PatternDetector
from primitives.common import EOSBaseModel

logger = structlog.get_logger("synapse.detectors.intent_rate_detector")

class IntentRatePattern(EOSBaseModel):
    """Representation of intent generation rate pattern."""
    cycles_below_threshold: int
    current_intent_rate: float
    minimum_intent_rate: float
    incident_id: str | None = None

class IntentRateDetector(PatternDetector):
    """
    Detect cognitive stalls by monitoring intent generation rate.
    
    Tracks intent generation rate over a sliding window and identifies
    sustained performance drops.
    """
    
    @classmethod
    def scan(
        cls, 
        window: List[Dict[str, Any]], 
        config: Dict[str, Any]
    ) -> List[IntentRatePattern]:
        """
        Scan for intent rate anomalies.
        
        Args:
            window: Historical performance data
            config: Detection configuration parameters
        
        Returns:
            List of detected intent rate patterns
        """
        minimum_intent_rate = config.get('minimum_intent_rate', 0.5)
        detection_window = config.get('detection_window', 10)
        
        detected_patterns: List[IntentRatePattern] = []
        
        # Compute sliding window statistics
        recent_rates = [entry.get('intent_rate', 0) for entry in window[-detection_window:]]
        
        if len(recent_rates) < detection_window:
            return detected_patterns
        
        # Count cycles below threshold
        cycles_below = sum(1 for rate in recent_rates if rate < minimum_intent_rate)
        
        if cycles_below >= detection_window * 0.7:  # 70% of window
            pattern = IntentRatePattern(
                cycles_below_threshold=cycles_below,
                current_intent_rate=recent_rates[-1],
                minimum_intent_rate=minimum_intent_rate,
                incident_id=f"STALL-{cls._generate_incident_id()}"
            )
            
            logger.warning(
                "Cognitive stall detected",
                cycles_below=cycles_below,
                current_rate=pattern.current_intent_rate
            )
            
            detected_patterns.append(pattern)
        
        return detected_patterns
    
    @staticmethod
    def _generate_incident_id() -> str:
        """Generate a unique incident identifier."""
        import uuid
        return str(uuid.uuid4())[:8].upper()