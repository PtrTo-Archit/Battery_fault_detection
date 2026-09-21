"""Unit tests for src/validation.py

Bounds are calibrated to the real dataset (discharge-only, current
always negative), so test fixtures use realistic discharge current
values (negative) rather than arbitrary positive numbers.
"""

from src.validation import Reading, validate_batch, validate_reading


class TestValidateReading:
    def test_valid_reading_passes(self):
        valid, reason = validate_reading(3.7, -2.0, 25.0)
        assert valid is True
        assert reason == ""

    def test_overvoltage_rejected(self):
        valid, reason = validate_reading(6.5, -2.0, 25.0)
        assert valid is False
        assert "Voltage" in reason

    def test_negative_voltage_rejected(self):
        valid, reason = validate_reading(-1.0, -2.0, 25.0)
        assert valid is False
        assert "Voltage" in reason

    def test_current_out_of_range_rejected(self):
        valid, reason = validate_reading(3.7, 999.0, 25.0)
        assert valid is False
        assert "Current" in reason

    def test_current_too_negative_rejected(self):
        # Below the discharge-only lower bound (-6.0A)
        valid, reason = validate_reading(3.7, -10.0, 25.0)
        assert valid is False
        assert "Current" in reason

    def test_extreme_temperature_rejected(self):
        valid, reason = validate_reading(3.7, -2.0, 500.0)
        assert valid is False
        assert "Temperature" in reason

    def test_boundary_values_are_valid(self):
        # Boundaries themselves should be inclusive/valid, not rejected
        valid, reason = validate_reading(2.0, -6.0, -10.0)
        assert valid is True

    def test_missing_field_rejected(self):
        valid, reason = validate_reading(None, -2.0, 25.0)
        assert valid is False
        assert "Missing" in reason


class TestValidateBatch:
    def test_mixed_batch_splits_correctly(self):
        readings = [
            Reading(voltage=3.7, current=-2.0, temperature=25.0),  # valid
            Reading(voltage=99.0, current=-2.0, temperature=25.0),  # invalid voltage
            Reading(voltage=3.7, current=-2.0, temperature=25.0),  # valid
        ]
        valid, rejected = validate_batch(readings)
        assert len(valid) == 2
        assert len(rejected) == 1

    def test_empty_batch(self):
        valid, rejected = validate_batch([])
        assert valid == []
        assert rejected == []

    def test_all_invalid_batch(self):
        readings = [Reading(voltage=99.0, current=-2.0, temperature=25.0)]
        valid, rejected = validate_batch(readings)
        assert len(valid) == 0
        assert len(rejected) == 1
