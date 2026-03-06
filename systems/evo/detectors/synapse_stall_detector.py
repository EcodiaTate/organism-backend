from __future__ import annotations

import structlog
from typing import Any, Dict, Optional

from primitives.common import EOSBaseModel
from systems.evo.detectors import PatternDetector
from systems.axon.types import DetectionResult

logger = structlog.get_logger("synapse_stall_detector")

class SynapseStallDetectorInput(EOSBaseModel):
    broadcast_ack_rate: float
    cycle_count: int

class SynapseStallDetector(PatternDetector):
    """
    Detects cognitive stalls in Synapse by monitoring broadcast acknowledgement rates.
    
    Stall Criteria:
    - Broadcast acknowledgement rate below 0.3
    - Sustained over multiple cycles
    """
    
    pattern_type: str = "synapse_cognitive_stall"
    description: str = "Detect cognitive coordination failures in Synapse"

    def scan(self, data: Dict[str, Any]) -> Optional[DetectionResult]:
        """
        Scan for cognitive stall conditions in Synapse.
        
        Args:
            data: Input data containing broadcast acknowledgement metrics
        
        Returns:
            DetectionResult if a stall is detected, None otherwise
        """
        try:
            input_data = SynapseStallDetectorInput.model_validate(data)
            
            if (input_data.broadcast_ack_rate < 0.3 and 
                input_data.cycle_count >= 50):
                
                logger.warning(
                    "Cognitive stall detected", 
                    broadcast_ack_rate=input_data.broadcast_ack_rate,
                    cycle_count=input_data.cycle_count
                )
                
                return DetectionResult(
                    detected=True,
                    severity=0.8,  # High severity
                    pattern_type=self.pattern_type,
                    details={
                        "broadcast_ack_rate": input_data.broadcast_ack_rate,
                        "cycle_count": input_data.cycle_count
                    }
                )
            
            return None
        
        except Exception as e:
            logger.error("Error in stall detection", error=str(e))
            return None