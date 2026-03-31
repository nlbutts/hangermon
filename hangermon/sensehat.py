"""Sense HAT interface for temperature, humidity, and LED control."""
from __future__ import annotations

import logging
import threading
from typing import Optional

LOGGER = logging.getLogger(__name__)

try:
    from sense_hat import SenseHat as _SenseHat  # type: ignore
    _HAS_SENSE_HAT = True
except ImportError:
    _HAS_SENSE_HAT = False
    LOGGER.warning("sense_hat library not available; Sense HAT features disabled")


class SenseHatController:
    """Thread-safe wrapper around the Sense HAT hardware."""

    def __init__(self) -> None:
        self._hat: Optional[object] = None
        self._lock = threading.Lock()
        self._led_intensity: int = 0
        self._temperature: float = 0.0
        self._humidity: float = 0.0

        if _HAS_SENSE_HAT:
            try:
                self._hat = _SenseHat()
                self._hat.clear(0, 0, 0)
                self._hat.low_light = True
                LOGGER.info("Sense HAT initialized successfully")
            except Exception:
                LOGGER.warning("Failed to initialize Sense HAT", exc_info=True)
                self._hat = None

    def read_sensors(self) -> dict:
        """Read temperature and humidity. Safe to call from any thread."""
        if self._hat is None:
            return {"temperature_c": 0.0, "temperature_f": 32.0, "humidity": 0.0}

        with self._lock:
            try:
                self._temperature = round(self._hat.get_temperature(), 1)
                self._humidity = round(self._hat.get_humidity(), 1)
            except Exception:
                LOGGER.warning("Failed to read Sense HAT sensors", exc_info=True)

        temp_f = round(self._temperature * 9.0 / 5.0 + 32.0, 1)
        return {
            "temperature_c": self._temperature,
            "temperature_f": temp_f,
            "humidity": self._humidity,
        }

    def set_led_intensity(self, intensity: int) -> None:
        """Set all LEDs to white at the given intensity (0-255)."""
        intensity = max(0, min(255, intensity))
        self._led_intensity = intensity

        if self._hat is None:
            return

        with self._lock:
            try:
                if intensity == 0:
                    self._hat.clear(0, 0, 0)
                else:
                    self._hat.clear(intensity, intensity, intensity)
            except Exception:
                LOGGER.warning("Failed to set Sense HAT LEDs", exc_info=True)

    def get_led_intensity(self) -> int:
        return self._led_intensity

    def shutdown(self) -> None:
        """Turn off LEDs on shutdown."""
        if self._hat is not None:
            with self._lock:
                try:
                    self._hat.clear(0, 0, 0)
                except Exception:
                    pass


# Module-level singleton
sensehat = SenseHatController()

__all__ = ["SenseHatController", "sensehat"]
