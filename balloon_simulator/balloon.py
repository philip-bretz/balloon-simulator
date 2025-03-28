from dataclasses import dataclass
from math import pi
from typing import TypeVar

import numpy as np

from balloon_simulator.constants import (
    GRAVITY,
    HELIUM_MOLAR_MASS,
    UNIVERSAL_GAS_CONSTANT,
)

T = TypeVar("T", float, np.ndarray)

G_TO_KG = 0.001


@dataclass
class SphericalBalloon:
    """
    Spherical balloon

    See https://nearspacelaunch.com/wp-content/uploads/2024/07/High_Altitude_Ballon_flight_fundamentals.pdf
    and https://www.sciencedirect.com/science/article/pii/S2214157X2301239X for reference equations

    burst_radius : float
        Burst radius [m]
    balloon_mass : float
        Mass of balloon film [kg]
    payload_mass : float
        Mass of payload and attachments [kg]
    drag_coefficient : float
        Coefficient of drag [unitless]
    virtual_mass_coefficient : float
        Coefficient of virtual mass [unitless]
    gas_molar_mass : float
        Molar mass of balloon gas [g/mol]
    """

    burst_radius: float
    balloon_mass: float
    payload_mass: float = 0.0
    drag_coefficient: float = 0.31
    virtual_mass_coefficient: float = 0.37
    gas_molar_mass: float = HELIUM_MOLAR_MASS

    def calculate_upward_acceleration(
        self, radius: float, speed: float, gas_mass: float, air_density: float
    ) -> float:
        """
        Calculate upward acceleration [m/s^2] from volume [m^3], speed [m/s], gas mass [kg], and air density [kg/m^3]

        See https://www.sciencedirect.com/science/article/pii/S2214157X2301239X
        """

        # Spherical asssumption, V = (4 / 3) * pi * r^3 and A = pi * r^2
        # radius = (volume / ((4 / 3) * pi)) ** (1 / 3)
        volume = (4 / 3) * pi * radius**3
        area = pi * radius**2

        # Buoyant force: F_B = (m_a - m_g) * g
        air_mass = air_density * volume
        buoyant_force = (air_mass - gas_mass) * GRAVITY

        # Drag force: F_D = (1 / 2) * C_D * A * rho_a * v^2
        drag_force = (1 / 2) * self.drag_coefficient * air_density * area * speed**2

        # Weight force: F_W = (m_b + m_p + m_g) * g
        weight_force = (self.balloon_mass + self.payload_mass + gas_mass) * GRAVITY

        # Net upward force: F = F_B - F_D - F_W (when going up) or F = F_B + F_D - F_W (when going down)
        if speed > 0:
            net_force = buoyant_force - weight_force - drag_force
        else:
            net_force = buoyant_force - weight_force + drag_force

        # Virtual mass: m_v = m_b + m_p + m_g + C_v * (rho_a * V)
        virtual_mass = (
            self.balloon_mass
            + self.payload_mass
            + gas_mass
            + self.virtual_mass_coefficient * air_mass
        )

        return net_force / virtual_mass

    def calculate_gas_mass_from_volume(
        self, volume: T, pressure: T, temperature: T
    ) -> T:
        """
        Calculate mass [kg] of gas from volume [m^3], pressure [Pa], and internal temperature [K] using
        the ideal gas law, PV = nRT
        """
        molar_amount = pressure * volume / (UNIVERSAL_GAS_CONSTANT * temperature)
        return molar_amount * self.gas_molar_mass * G_TO_KG

    def calculate_volume_from_gas_mass(
        self, gas_mass: T, pressure: T, temperature: T
    ) -> T:
        """
        Calculate volume [m^3] from mass [kg] of gas, pressure [Pa], and internal temperature [K] using
        the ideal gas law, PV = nRT
        """
        molar_amount = gas_mass / (self.gas_molar_mass * G_TO_KG)
        return molar_amount * UNIVERSAL_GAS_CONSTANT * temperature / pressure
