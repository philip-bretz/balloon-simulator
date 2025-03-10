from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtmosphereQuery:
    latitude: np.ndarray
    longitude: np.ndarray
    altitude: np.ndarray
    time: np.ndarray


@dataclass(frozen=True)
class AtmosphereState:
    temperature: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    u_wind: np.ndarray
    v_wind: np.ndarray
    w_wind: np.ndarray
