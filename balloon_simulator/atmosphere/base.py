from abc import ABC, abstractmethod

import numpy as np

from balloon_simulator.atmosphere.state import (
    AtmosphereArrayQuery,
    AtmosphereArrayState,
    AtmosphereState,
)


class AtmosphereBase(ABC):
    @abstractmethod
    def _calculate_state_core(
        self, query: AtmosphereArrayQuery
    ) -> AtmosphereArrayState: ...

    def calculate_state(
        self, latitude: float, longitude: float, altitude: float, time: float
    ) -> AtmosphereState:
        query = AtmosphereArrayQuery.from_floats(latitude, longitude, altitude, time)
        array_state = self._calculate_state_core(query)
        return AtmosphereState.from_array_state(array_state)

    def calculate_state_v(
        self,
        latitude: np.ndarray,
        longitude: np.ndarray,
        altitude: np.ndarray,
        time: np.ndarray,
    ) -> AtmosphereArrayState:
        query = AtmosphereArrayQuery(
            latitude=latitude, longitude=longitude, altitude=altitude, time=time
        )
        return self._calculate_state_core(query)
