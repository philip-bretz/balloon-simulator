from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtmosphereArrayQuery:
    latitude: np.ndarray
    longitude: np.ndarray
    altitude: np.ndarray
    time: np.ndarray

    @classmethod
    def from_floats(
        cls, latitude: float, longitude: float, altitude: float, time: float
    ):
        return cls(
            latitude=np.array([latitude]),
            longitude=np.array([longitude]),
            altitude=np.array([altitude]),
            time=np.array([time]),
        )


@dataclass(frozen=True)
class AtmosphereArrayState:
    temperature: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    u_wind: np.ndarray
    v_wind: np.ndarray
    w_wind: np.ndarray


@dataclass(frozen=True)
class AtmosphereState:
    temperature: float
    pressure: float
    density: float
    u_wind: float
    v_wind: float
    w_wind: float

    @classmethod
    def from_array_state(cls, array_state: AtmosphereArrayState, index: int = 0):
        return cls(
            temperature=array_state.temperature[index],
            pressure=array_state.pressure[index],
            density=array_state.density[index],
            u_wind=array_state.u_wind[index],
            v_wind=array_state.v_wind[index],
            w_wind=array_state.w_wind[index],
        )
