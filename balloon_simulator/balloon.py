from dataclasses import dataclass
from math import pi
from typing import TypeVar

import numpy as np

from balloon_simulator.constants import GRAVITY

T = TypeVar("T", float, np.ndarray)


@dataclass(frozen=True)
class SphericalBalloon:
    """
    Spherical balloon
    """

    burst_diameter: float
    balloon_mass: float
    payload_mass: float = 0.0
    drag_coefficient: float = 0.35

    def total_mass(self, volume: T, gas_density: T) -> T:
        return self.balloon_mass + self.payload_mass + gas_density * volume

    def buoyancy_force(self, volume: T, air_density: T) -> T:
        return air_density * volume * GRAVITY

    def gravity_force(self, volume: T, gas_density: T) -> T:
        return self.total_mass(volume, gas_density) * GRAVITY

    def drag_force(self, area: T, speed: T, air_density: T) -> T:
        return (1 / 2) * self.drag_coefficient * air_density * area * speed**2

    def acceleration(self, diameter: T, speed: T, gas_density: T, air_density: T) -> T:
        radius = diameter / 2.0
        volume = (4 / 3) * pi * radius**3
        area = pi * radius**2
        buoyant_force = self.buoyancy_force(volume, air_density)
        gravity_force = self.gravity_force(volume, gas_density)
        drag_force = self.drag_force(area, speed, air_density)
        net_force = buoyant_force - gravity_force - drag_force
        total_mass = self.total_mass(volume, gas_density)
        return net_force / total_mass
