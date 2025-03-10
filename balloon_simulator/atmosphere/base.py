from abc import ABC, abstractmethod

from balloon_simulator.atmosphere.state import AtmosphereQuery, AtmosphereState


class AtmosphereBase(ABC):
    @abstractmethod
    def calculate_state(self, query: AtmosphereQuery) -> AtmosphereState: ...
