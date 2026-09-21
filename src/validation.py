"""
validation.py
--------------
Input validation layer for battery telemetry readings.

Rejects physically impossible or out-of-range sensor values before
they reach the feature engineering / model layers. This exists as a
separate module (rather than inline checks) so it can be unit tested
in isolation and reused by both the batch pipeline and the live API.
"""

from dataclasses import dataclass


# Physical bounds for a single Li-ion cell, calibrated against the real
# battery_health_dataset.csv (NASA PCoE discharge data) plus a safety
# margin, rather than generic placeholder numbers:
#   observed voltage:      2.73V - 4.23V   -> bounds: 2.0 - 4.3V
#   observed current:      -4.03A - -0.89A -> bounds: -6.0 - 1.0A
#   observed temperature:  4.7C - 66.7C    -> bounds: -10 - 80C
#
# Note this dataset is discharge-only, so current is always negative;
# the small positive headroom (up to 1.0A) allows for near-zero/charging
# readings if you later combine this with a charging dataset.
VOLTAGE_MIN, VOLTAGE_MAX = 2.0, 4.3        # volts
CURRENT_MIN, CURRENT_MAX = -6.0, 1.0       # amps (negative = discharge)
TEMP_MIN, TEMP_MAX = -10.0, 80.0           # degrees Celsius


@dataclass
class Reading:
    """A single battery telemetry sample."""
    voltage: float
    current: float
    temperature: float
    timestamp: float = 0.0  # seconds, optional


def validate_reading(voltage: float, current: float, temperature: float) -> tuple[bool, str]:
    """
    Validates a single battery telemetry reading against physical bounds.

    Returns:
        (is_valid, reason) — reason is an empty string when valid,
        otherwise a short human-readable explanation of the violation.
    """
    if voltage is None or current is None or temperature is None:
        return False, "Missing field(s) in reading"

    if not (VOLTAGE_MIN <= voltage <= VOLTAGE_MAX):
        return False, f"Voltage {voltage}V out of physical range [{VOLTAGE_MIN}, {VOLTAGE_MAX}]"

    if not (CURRENT_MIN <= current <= CURRENT_MAX):
        return False, f"Current {current}A out of physical range [{CURRENT_MIN}, {CURRENT_MAX}]"

    if not (TEMP_MIN <= temperature <= TEMP_MAX):
        return False, f"Temperature {temperature}C out of physical range [{TEMP_MIN}, {TEMP_MAX}]"

    return True, ""


def validate_batch(readings: list[Reading]) -> tuple[list[Reading], list[tuple[Reading, str]]]:
    """
    Validates a list of readings.

    Returns:
        (valid_readings, rejected) where rejected is a list of
        (reading, reason) tuples for anything that failed validation.
    """
    valid: list[Reading] = []
    rejected: list[tuple[Reading, str]] = []

    for r in readings:
        is_valid, reason = validate_reading(r.voltage, r.current, r.temperature)
        if is_valid:
            valid.append(r)
        else:
            rejected.append((r, reason))

    return valid, rejected
