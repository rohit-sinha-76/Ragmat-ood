"""Test Anomaly Logger."""
import os
import json
import pytest
from pathlib import Path
from ragmat.logging_utils import AnomalyLogger, _LOGS_DIR

def test_critical_failure_raises():
    """Verify critical failure raises RuntimeError and writes log."""
    with pytest.raises(RuntimeError, match="CRITICAL FAILURE -- see logs/critical_failures.log: test_fail"):
        AnomalyLogger.log_critical_failure("test_fail", {"a": 1})
    
    log_path = _LOGS_DIR / 'critical_failures.log'
    assert log_path.exists()
    
    with open(log_path) as f:
        lines = f.readlines()
    
    # Must be valid JSON and contain the message
    entry = json.loads(lines[-1])
    assert entry["msg"] == "test_fail"
    assert entry["data"]["a"] == 1
    assert entry["type"] == "CRITICAL_FAILURE"
