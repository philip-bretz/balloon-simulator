from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from balloon_simulator.atmosphere.base import AtmosphereBase
from balloon_simulator.atmosphere.state import AtmosphereQuery, AtmosphereState
from balloon_simulator.constants import GRAVITY, UNIVERSAL_AIR_CONSTANT

MSL_TEMPERATURE = 288.15  # K
MSL_PRESSURE = 1013.25  # hPa
MSL_DENSITY = 1.225  # kg/m^3

TROPOSPHERE_ALTITUDE = 0.0  # m
TROPOPAUSE_ALTITUDE = 11_000.0  # m
STRATOSPHERE1_ALTITUDE = 20_000.0  # m
STRATOSPHERE2_ALTITUDE = 32_000.0  # m

TROPOSPHERE_LAPSE_RATE = -0.0065  # K/m
STRATOSPHERE1_LAPSE_RATE = 0.001  # K/m


T = TypeVar("T", float, np.ndarray)


@dataclass(frozen=True)
class ISAMasks:
    troposphere: np.ndarray
    tropopause: np.ndarray
    stratosphere1: np.ndarray

    @classmethod
    def from_altitude(cls, altitude: np.ndarray):
        troposphere_mask = (altitude >= TROPOSPHERE_ALTITUDE) & (
            altitude <= TROPOPAUSE_ALTITUDE
        )
        tropopause_mask = (altitude >= TROPOPAUSE_ALTITUDE) & (
            altitude <= STRATOSPHERE1_ALTITUDE
        )
        stratosphere1_mask = (altitude >= STRATOSPHERE1_ALTITUDE) & (
            altitude <= STRATOSPHERE2_ALTITUDE
        )
        return cls(
            troposphere=troposphere_mask,
            tropopause=tropopause_mask,
            stratosphere1=stratosphere1_mask,
        )


class ISAAtmosphere(AtmosphereBase):
    def __init__(self, offset: float = 0.0):
        self._offset = offset

    def __repr__(self) -> str:
        return f"ISAAtmosphere(offset={self._offset})"

    @property
    def _msl_temperature(self) -> float:
        return MSL_TEMPERATURE + self._offset

    def calculate_state(self, query: AtmosphereQuery) -> AtmosphereState:
        return self._calculate_state_for_altitude(query.altitude)

    def _calculate_state_for_altitude(self, altitude: np.ndarray) -> AtmosphereState:
        temperature = self.temperature_at_altitude(altitude)
        pressure = self.pressure_at_altitude(altitude)
        density = self.density_at_altitude(altitude)
        return AtmosphereState(
            temperature=temperature,
            pressure=pressure,
            density=density,
            u_wind=np.zeros_like(altitude),
            v_wind=np.zeros_like(altitude),
            w_wind=np.zeros_like(altitude),
        )

    def _troposphere_temperature(self, altitude: T) -> T:
        return self._msl_temperature + TROPOSPHERE_LAPSE_RATE * (
            altitude - TROPOSPHERE_ALTITUDE
        )

    @property
    def _tropopause_temperature(self) -> float:
        return self._troposphere_temperature(TROPOPAUSE_ALTITUDE)

    def _stratosphere1_temperature(self, altitude: T) -> T:
        return self._tropopause_temperature + STRATOSPHERE1_LAPSE_RATE * (
            altitude - STRATOSPHERE1_ALTITUDE
        )

    def temperature_at_altitude(self, altitude: np.ndarray) -> np.ndarray:
        """
        Calculate temperature [K] at altitude [m], vectorized

        Calculation is split between troposphere, tropopause, and first part of stratosphere,
        with NaN for altitudes outside that range.
        """
        temperature = np.full(altitude.shape, np.nan)
        masks = ISAMasks.from_altitude(altitude)
        temperature[masks.troposphere] = self._troposphere_temperature(
            altitude[masks.troposphere]
        )
        temperature[masks.tropopause] = self._tropopause_temperature
        temperature[masks.stratosphere1] = self._stratosphere1_temperature(
            altitude[masks.stratosphere1]
        )
        return temperature

    def _troposphere_pressure(self, altitude: T) -> T:
        return MSL_PRESSURE * (
            1 + (TROPOSPHERE_LAPSE_RATE * altitude / self._msl_temperature)
        ) ** (-GRAVITY / (TROPOSPHERE_LAPSE_RATE * UNIVERSAL_AIR_CONSTANT))

    @property
    def _tropopause_start_pressure(self) -> float:
        return self._troposphere_pressure(TROPOPAUSE_ALTITUDE)

    def _tropopause_pressure(self, altitude: T) -> T:
        return self._tropopause_start_pressure * np.exp(
            -GRAVITY
            / (UNIVERSAL_AIR_CONSTANT * self._tropopause_temperature)
            * (altitude - TROPOPAUSE_ALTITUDE)
        )

    @property
    def _stratosphere1_start_pressure(self) -> float:
        return self._tropopause_pressure(STRATOSPHERE1_ALTITUDE)

    def _stratosphere1_pressure(self, altitude: T) -> T:
        return self._stratosphere1_start_pressure * (
            1
            + STRATOSPHERE1_LAPSE_RATE
            * (altitude - STRATOSPHERE1_ALTITUDE)
            / self._tropopause_temperature
        ) ** (-GRAVITY / (STRATOSPHERE1_LAPSE_RATE * UNIVERSAL_AIR_CONSTANT))

    def pressure_at_altitude(self, altitude: np.ndarray) -> np.ndarray:
        """
        Calculate pressure [hPa] at altitude [m], vectorized

        Calculation is split between troposphere, tropopause, and first part of stratosphere,
        with NaN for altitudes outside that range.
        """
        pressure = np.full(altitude.shape, np.nan)
        masks = ISAMasks.from_altitude(altitude)
        pressure[masks.troposphere] = self._troposphere_pressure(
            altitude[masks.troposphere]
        )
        pressure[masks.tropopause] = self._tropopause_pressure(
            altitude[masks.tropopause]
        )
        pressure[masks.stratosphere1] = self._stratosphere1_pressure(
            altitude[masks.stratosphere1]
        )
        return pressure

    def density_at_altitude(self, altitude: np.ndarray) -> np.ndarray:
        """
        Calculate pressure [hPa] at altitude [m], vectorized

        Calculation is split between troposphere, tropopause, and first part of stratosphere,
        with NaN for altitudes outside that range.
        """
        raise NotImplementedError
