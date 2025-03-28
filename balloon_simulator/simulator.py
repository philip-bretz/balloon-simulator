from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import ceil

import numpy as np

from balloon_simulator.atmosphere import AtmosphereBase, ISAAtmosphere
from balloon_simulator.balloon import SphericalBalloon


class NumericalMethod(StrEnum):
    EULER = "euler"
    RUNGE_KUTTA_4 = "runge_kutta_4"


@dataclass(frozen=True)
class SimulationSettings:
    time_max: float
    timestep: float = 1.0
    numerical_method: NumericalMethod = NumericalMethod.RUNGE_KUTTA_4
    verbose: bool = False


@dataclass(frozen=True)
class SimulationState:
    altitude: float
    speed: float


@dataclass(frozen=True)
class SimulationStateDelta:
    altitude: float
    speed: float

    def __radd__(self, other: SimulationState) -> SimulationState:
        return SimulationState(
            altitude=self.altitude + other.altitude,
            speed=self.speed + other.speed,
        )

    def __add__(self, other: SimulationStateDelta) -> SimulationStateDelta:
        return SimulationStateDelta(
            altitude=self.altitude + other.altitude,
            speed=self.speed + other.speed,
        )

    def __rmul__(self, scalar: float) -> SimulationStateDelta:
        return SimulationStateDelta(
            altitude=scalar * self.altitude,
            speed=scalar * self.speed,
        )


@dataclass(frozen=True)
class SimulationResults:
    time: np.ndarray
    altitude: np.ndarray
    speed: np.ndarray
    acceleration: np.ndarray
    radius: np.ndarray


@dataclass(frozen=True)
class BalloonSimulator:
    balloon: SphericalBalloon
    atmosphere: AtmosphereBase = field(default_factory=ISAAtmosphere)

    def simulate(
        self,
        initial_time: float,
        initial_state: SimulationState,
        gas_mass: float,
        sim_settings: SimulationSettings,
    ) -> SimulationResults:
        num_steps = ceil((sim_settings.time_max - initial_time) / sim_settings.timestep)
        states_and_times = [(initial_state, initial_time)]
        for _ in range(num_steps):
            state, time = states_and_times[-1]
            new_state = self.step(
                time=time,
                state=state,
                dt=sim_settings.timestep,
                method=sim_settings.numerical_method,
                gas_mass=gas_mass,
            )
            new_time = time + sim_settings.timestep
            states_and_times.append((new_state, new_time))
            if sim_settings.verbose:
                print(f"{new_time}: {new_state}")
        return self._states_to_results(states_and_times, gas_mass=gas_mass)

    def _states_to_results(
        self, states_and_times: list[tuple[SimulationState, float]], gas_mass: float
    ) -> SimulationResults:
        time = np.array([t for _, t in states_and_times])
        altitude = np.array([state.altitude for state, _ in states_and_times])
        speed = np.array([state.speed for state, _ in states_and_times])
        atmosphere_state = self.atmosphere.calculate_state_v(
            np.zeros_like(altitude), np.zeros_like(altitude), altitude, time
        )
        volume = self.balloon.calculate_volume_from_gas_mass(  # type: ignore
            gas_mass, atmosphere_state.pressure, atmosphere_state.temperature
        )
        radius = (volume / ((4 / 3) * np.pi)) ** (1 / 3)  # type: ignore
        acceleration = np.array(
            [
                self.balloon.calculate_upward_acceleration(r, v, gas_mass, rho)
                for r, v, rho in zip(radius, speed, atmosphere_state.density)
            ]
        )
        return SimulationResults(
            time=time,
            altitude=altitude,
            speed=speed,
            acceleration=acceleration,
            radius=radius,
        )

    def step(
        self,
        time: float,
        state: SimulationState,
        dt: float,
        method: NumericalMethod,
        gas_mass: float,
    ) -> SimulationState:
        if method == NumericalMethod.EULER:
            return self._euler(time=time, state=state, dt=dt, gas_mass=gas_mass)
        elif method == NumericalMethod.RUNGE_KUTTA_4:
            return self._runge_kutta_4(time=time, state=state, dt=dt, gas_mass=gas_mass)
        else:
            raise NotImplementedError

    def _euler(
        self, time: float, state: SimulationState, dt: float, **kwargs
    ) -> SimulationState:
        state_dot = self.state_dot(time, state, **kwargs)
        return state + dt * state_dot

    def _runge_kutta_4(
        self, time: float, state: SimulationState, dt: float, **kwargs
    ) -> SimulationState:
        k_1 = self.state_dot(time, state, **kwargs)
        k_2 = self.state_dot(time + dt / 2, state + (dt / 2) * k_1, **kwargs)
        k_3 = self.state_dot(time + dt / 2, state + (dt / 2) * k_2, **kwargs)
        k_4 = self.state_dot(time + dt, state + dt * k_3, **kwargs)
        return state + (dt / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

    def state_dot(
        self, time: float, state: SimulationState, gas_mass: float
    ) -> SimulationStateDelta:
        atmosphere_state = self.atmosphere.calculate_state(
            0.0, 0.0, state.altitude, time
        )
        volume = self.balloon.calculate_volume_from_gas_mass(
            gas_mass, atmosphere_state.pressure, atmosphere_state.temperature
        )
        radius = (volume / ((4 / 3) * np.pi)) ** (1 / 3)
        acceleration = self.balloon.calculate_upward_acceleration(
            radius, state.speed, gas_mass, atmosphere_state.density
        )
        return SimulationStateDelta(altitude=state.speed, speed=acceleration)
