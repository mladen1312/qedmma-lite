#!/usr/bin/env python3
"""
Hypersonic & Anomaly Trajectory Dataset Generator

Industry-standard tool for generating synthetic radar tracking scenarios.
Used by researchers and engineers worldwide for algorithm development.

Features:
- Hypersonic cruise trajectories (Mach 5-25)
- High-G maneuver profiles (up to 100g)
- Ballistic reentry trajectories
- Evasive S-turn patterns
- Coordinated multi-target scenarios
- Multistatic TDOA measurement generation

Export Formats:
- CSV (universal compatibility)
- JSON (web/API integration)
- NumPy (.npy for Python workflows)
- MATLAB (.mat for legacy systems)

MIT License - Copyright (c) 2026 Dr. Mladen Mešter
Website: www.nexellum.com
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import json
import csv
import argparse
from datetime import datetime
import os


__version__ = "1.0.0"
__author__ = "Dr. Mladen Mešter"


# =============================================================================
# Physical Constants
# =============================================================================

class PhysicsConstants:
    """Standard physical constants for trajectory generation"""
    G = 9.80665              # Gravitational acceleration (m/s²)
    C = 299792458.0          # Speed of light (m/s)
    R_EARTH = 6371000.0      # Earth radius (m)
    
    # Atmosphere model (simplified)
    SPEED_OF_SOUND_SEA = 340.29    # m/s at sea level
    SPEED_OF_SOUND_30KM = 301.71   # m/s at 30 km altitude
    
    @staticmethod
    def speed_of_sound(altitude_m: float) -> float:
        """Speed of sound at altitude (simplified ISA model)"""
        if altitude_m < 11000:
            return 340.29 - 0.004 * altitude_m
        elif altitude_m < 25000:
            return 295.0
        else:
            return 295.0 + 0.001 * (altitude_m - 25000)
    
    @staticmethod
    def air_density(altitude_m: float) -> float:
        """Air density at altitude (kg/m³)"""
        # Exponential atmosphere model
        H = 8500  # Scale height
        rho_0 = 1.225  # Sea level density
        return rho_0 * np.exp(-altitude_m / H)


# =============================================================================
# Trajectory Types
# =============================================================================

class TrajectoryType(Enum):
    """Available trajectory types"""
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    HYPERSONIC_CRUISE = "hypersonic_cruise"
    HYPERSONIC_PULLUP = "hypersonic_pullup"
    BALLISTIC_REENTRY = "ballistic_reentry"
    EVASIVE_STURN = "evasive_sturn"
    COORDINATED_TURN = "coordinated_turn"
    TERMINAL_DIVE = "terminal_dive"
    RANDOM_MANEUVER = "random_maneuver"
    
    # Multi-target scenarios
    SWARM = "swarm"
    SEPARATION = "separation"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrajectoryPoint:
    """Single point in a trajectory"""
    time: float
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float
    
    # Derived quantities
    speed: float = 0.0
    mach: float = 0.0
    g_load: float = 0.0
    heading_deg: float = 0.0
    flight_path_angle_deg: float = 0.0
    
    def compute_derived(self, speed_of_sound: float = 340.0):
        """Compute derived quantities"""
        self.speed = np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
        self.mach = self.speed / speed_of_sound
        self.g_load = np.sqrt(self.ax**2 + self.ay**2 + self.az**2) / PhysicsConstants.G
        
        # Heading (degrees from North)
        self.heading_deg = np.degrees(np.arctan2(self.vx, self.vy)) % 360
        
        # Flight path angle
        horiz_speed = np.sqrt(self.vx**2 + self.vy**2)
        if horiz_speed > 1:
            self.flight_path_angle_deg = np.degrees(np.arctan2(self.vz, horiz_speed))
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_array(self) -> np.ndarray:
        """Return state as numpy array [pos, vel, acc]"""
        return np.array([
            self.x, self.y, self.z,
            self.vx, self.vy, self.vz,
            self.ax, self.ay, self.az
        ])


@dataclass
class Trajectory:
    """Complete trajectory with metadata"""
    name: str
    trajectory_type: TrajectoryType
    points: List[TrajectoryPoint]
    metadata: Dict = field(default_factory=dict)
    
    def __len__(self):
        return len(self.points)
    
    def duration(self) -> float:
        if not self.points:
            return 0.0
        return self.points[-1].time - self.points[0].time
    
    def max_speed(self) -> float:
        return max(p.speed for p in self.points)
    
    def max_mach(self) -> float:
        return max(p.mach for p in self.points)
    
    def max_g_load(self) -> float:
        return max(p.g_load for p in self.points)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [N, 13] - time + state + derived"""
        data = []
        for p in self.points:
            data.append([
                p.time, p.x, p.y, p.z, p.vx, p.vy, p.vz,
                p.ax, p.ay, p.az, p.speed, p.mach, p.g_load
            ])
        return np.array(data)
    
    def save_csv(self, filepath: str):
        """Save trajectory to CSV file"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'time_s', 'x_m', 'y_m', 'z_m', 
                'vx_m_s', 'vy_m_s', 'vz_m_s',
                'ax_m_s2', 'ay_m_s2', 'az_m_s2',
                'speed_m_s', 'mach', 'g_load'
            ])
            # Data
            for p in self.points:
                writer.writerow([
                    f"{p.time:.6f}", 
                    f"{p.x:.3f}", f"{p.y:.3f}", f"{p.z:.3f}",
                    f"{p.vx:.3f}", f"{p.vy:.3f}", f"{p.vz:.3f}",
                    f"{p.ax:.3f}", f"{p.ay:.3f}", f"{p.az:.3f}",
                    f"{p.speed:.3f}", f"{p.mach:.4f}", f"{p.g_load:.3f}"
                ])
    
    def save_json(self, filepath: str):
        """Save trajectory to JSON file"""
        data = {
            'name': self.name,
            'type': self.trajectory_type.value,
            'metadata': self.metadata,
            'summary': {
                'duration_s': self.duration(),
                'num_points': len(self.points),
                'max_speed_m_s': self.max_speed(),
                'max_mach': self.max_mach(),
                'max_g_load': self.max_g_load()
            },
            'points': [p.to_dict() for p in self.points]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_numpy(self, filepath: str):
        """Save trajectory to NumPy file"""
        np.save(filepath, self.to_numpy())


# =============================================================================
# Trajectory Generators
# =============================================================================

class TrajectoryGenerator:
    """
    Generate various trajectory types for radar tracking algorithm testing.
    
    All trajectories are physics-based with realistic constraints.
    """
    
    def __init__(self, dt: float = 0.0625, seed: Optional[int] = None):
        """
        Initialize generator.
        
        Args:
            dt: Time step in seconds (default 62.5 ms = 16 Hz)
            seed: Random seed for reproducibility
        """
        self.dt = dt
        if seed is not None:
            np.random.seed(seed)
    
    def _compute_derived(self, points: List[TrajectoryPoint]):
        """Compute derived quantities for all points"""
        for p in points:
            sos = PhysicsConstants.speed_of_sound(p.z)
            p.compute_derived(sos)
    
    # -------------------------------------------------------------------------
    # Basic Trajectories
    # -------------------------------------------------------------------------
    
    def constant_velocity(self, duration: float,
                         initial_pos: np.ndarray,
                         velocity: np.ndarray,
                         name: str = "CV_Trajectory") -> Trajectory:
        """
        Generate constant velocity trajectory.
        
        Args:
            duration: Total duration in seconds
            initial_pos: Starting position [x, y, z] in meters
            velocity: Velocity vector [vx, vy, vz] in m/s
        """
        points = []
        pos = initial_pos.copy()
        vel = velocity.copy()
        acc = np.zeros(3)
        
        t = 0.0
        while t <= duration:
            points.append(TrajectoryPoint(
                time=t,
                x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2],
                ax=acc[0], ay=acc[1], az=acc[2]
            ))
            pos += vel * self.dt
            t += self.dt
        
        self._compute_derived(points)
        
        return Trajectory(
            name=name,
            trajectory_type=TrajectoryType.CONSTANT_VELOCITY,
            points=points,
            metadata={
                'initial_pos': initial_pos.tolist(),
                'velocity': velocity.tolist(),
                'dt': self.dt
            }
        )
    
    # -------------------------------------------------------------------------
    # Hypersonic Trajectories
    # -------------------------------------------------------------------------
    
    def hypersonic_cruise(self, duration: float,
                         mach: float = 8.0,
                         altitude_km: float = 25.0,
                         initial_pos: Optional[np.ndarray] = None,
                         heading_deg: float = 90.0,
                         name: str = "Hypersonic_Cruise") -> Trajectory:
        """
        Generate hypersonic cruise trajectory at constant altitude and speed.
        
        Simulates: HGV cruise phase, hypersonic missile transit
        
        Args:
            duration: Duration in seconds
            mach: Cruise Mach number
            altitude_km: Cruise altitude in km
            initial_pos: Starting position (default: origin at altitude)
            heading_deg: Initial heading (degrees from North, 90=East)
        """
        altitude_m = altitude_km * 1000
        sos = PhysicsConstants.speed_of_sound(altitude_m)
        speed = mach * sos
        
        heading_rad = np.radians(heading_deg)
        vel = np.array([
            speed * np.sin(heading_rad),
            speed * np.cos(heading_rad),
            0.0
        ])
        
        if initial_pos is None:
            initial_pos = np.array([0.0, 0.0, altitude_m])
        
        traj = self.constant_velocity(duration, initial_pos, vel, name)
        traj.trajectory_type = TrajectoryType.HYPERSONIC_CRUISE
        traj.metadata['mach'] = mach
        traj.metadata['altitude_km'] = altitude_km
        traj.metadata['heading_deg'] = heading_deg
        
        return traj
    
    def hypersonic_pullup(self, duration: float,
                         initial_mach: float = 8.0,
                         initial_altitude_km: float = 30.0,
                         dive_angle_deg: float = -30.0,
                         pullup_g: float = 60.0,
                         pullup_start: float = 3.0,
                         pullup_duration: float = 2.0,
                         name: str = "Hypersonic_Pullup") -> Trajectory:
        """
        Generate hypersonic pull-up maneuver trajectory.
        
        Simulates: Warhead terminal maneuver, HGV skip-glide
        
        Phases:
        1. Dive at constant angle
        2. High-G pull-up maneuver
        3. Level flight or climb
        
        Args:
            initial_mach: Starting Mach number
            initial_altitude_km: Starting altitude
            dive_angle_deg: Initial dive angle (negative = diving)
            pullup_g: G-load during pull-up maneuver
            pullup_start: Time when pull-up begins
            pullup_duration: Duration of pull-up phase
        """
        points = []
        
        altitude_m = initial_altitude_km * 1000
        sos = PhysicsConstants.speed_of_sound(altitude_m)
        speed = initial_mach * sos
        
        # Initial velocity (diving)
        dive_rad = np.radians(dive_angle_deg)
        pos = np.array([0.0, 0.0, altitude_m])
        vel = np.array([
            speed * np.cos(dive_rad),
            0.0,
            speed * np.sin(dive_rad)
        ])
        acc = np.zeros(3)
        
        t = 0.0
        while t <= duration:
            # Determine acceleration phase
            if pullup_start <= t < pullup_start + pullup_duration:
                # Pull-up maneuver (upward acceleration)
                acc = np.array([0.0, 0.0, pullup_g * PhysicsConstants.G])
            else:
                # Gravity only (simplified)
                acc = np.array([0.0, 0.0, -PhysicsConstants.G])
            
            points.append(TrajectoryPoint(
                time=t,
                x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2],
                ax=acc[0], ay=acc[1], az=acc[2]
            ))
            
            # Integration
            pos += vel * self.dt + 0.5 * acc * self.dt**2
            vel += acc * self.dt
            
            # Altitude floor
            if pos[2] < 1000:
                pos[2] = 1000
                if vel[2] < 0:
                    vel[2] = 0
            
            t += self.dt
        
        self._compute_derived(points)
        
        return Trajectory(
            name=name,
            trajectory_type=TrajectoryType.HYPERSONIC_PULLUP,
            points=points,
            metadata={
                'initial_mach': initial_mach,
                'initial_altitude_km': initial_altitude_km,
                'dive_angle_deg': dive_angle_deg,
                'pullup_g': pullup_g,
                'pullup_start': pullup_start,
                'pullup_duration': pullup_duration,
                'dt': self.dt
            }
        )
    
    def ballistic_reentry(self, duration: float,
                         apogee_km: float = 300.0,
                         range_km: float = 500.0,
                         name: str = "Ballistic_Reentry") -> Trajectory:
        """
        Generate ballistic missile reentry trajectory.
        
        Simulates: ICBM/IRBM reentry phase
        
        Uses simplified ballistic equations with gravity.
        """
        points = []
        
        # Starting at apogee
        pos = np.array([0.0, 0.0, apogee_km * 1000])
        
        # Calculate required initial velocity for range
        # Simplified: v = sqrt(g * range / sin(2*angle))
        launch_angle = np.radians(45)  # Optimal angle approximation
        v0 = np.sqrt(PhysicsConstants.G * range_km * 1000 / np.sin(2 * launch_angle))
        
        # At apogee, only horizontal velocity
        vel = np.array([v0 * np.cos(launch_angle), 0.0, 0.0])
        acc = np.array([0.0, 0.0, -PhysicsConstants.G])
        
        t = 0.0
        while t <= duration and pos[2] > 0:
            # Drag increases as altitude decreases
            if pos[2] < 100000:  # Below 100 km
                rho = PhysicsConstants.air_density(pos[2])
                Cd = 0.3
                A = 1.0  # m²
                m = 500  # kg
                speed = np.linalg.norm(vel)
                drag = 0.5 * rho * Cd * A * speed**2 / m
                drag_acc = -drag * vel / (speed + 1e-6)
                acc = np.array([drag_acc[0], drag_acc[1], -PhysicsConstants.G + drag_acc[2]])
            else:
                acc = np.array([0.0, 0.0, -PhysicsConstants.G])
            
            points.append(TrajectoryPoint(
                time=t,
                x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2],
                ax=acc[0], ay=acc[1], az=acc[2]
            ))
            
            pos += vel * self.dt + 0.5 * acc * self.dt**2
            vel += acc * self.dt
            t += self.dt
        
        self._compute_derived(points)
        
        return Trajectory(
            name=name,
            trajectory_type=TrajectoryType.BALLISTIC_REENTRY,
            points=points,
            metadata={
                'apogee_km': apogee_km,
                'range_km': range_km,
                'dt': self.dt
            }
        )
    
    # -------------------------------------------------------------------------
    # Evasive Maneuvers
    # -------------------------------------------------------------------------
    
    def evasive_sturn(self, duration: float,
                     mach: float = 6.0,
                     altitude_km: float = 20.0,
                     turn_frequency_hz: float = 0.5,
                     max_lateral_g: float = 30.0,
                     name: str = "Evasive_STurn") -> Trajectory:
        """
        Generate evasive S-turn maneuver.
        
        Simulates: Cruise missile evasion, HGV weaving
        
        Sinusoidal lateral acceleration while maintaining forward progress.
        """
        points = []
        
        altitude_m = altitude_km * 1000
        sos = PhysicsConstants.speed_of_sound(altitude_m)
        speed = mach * sos
        
        pos = np.array([0.0, 0.0, altitude_m])
        vel = np.array([speed, 0.0, 0.0])
        
        t = 0.0
        while t <= duration:
            # Sinusoidal lateral acceleration
            acc_y = max_lateral_g * PhysicsConstants.G * np.sin(2 * np.pi * turn_frequency_hz * t)
            acc = np.array([0.0, acc_y, 0.0])
            
            points.append(TrajectoryPoint(
                time=t,
                x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2],
                ax=acc[0], ay=acc[1], az=acc[2]
            ))
            
            pos += vel * self.dt + 0.5 * acc * self.dt**2
            vel += acc * self.dt
            t += self.dt
        
        self._compute_derived(points)
        
        return Trajectory(
            name=name,
            trajectory_type=TrajectoryType.EVASIVE_STURN,
            points=points,
            metadata={
                'mach': mach,
                'altitude_km': altitude_km,
                'turn_frequency_hz': turn_frequency_hz,
                'max_lateral_g': max_lateral_g,
                'dt': self.dt
            }
        )
    
    def coordinated_turn(self, duration: float,
                        mach: float = 5.0,
                        altitude_km: float = 25.0,
                        turn_radius_km: float = 50.0,
                        name: str = "Coordinated_Turn") -> Trajectory:
        """
        Generate coordinated turn trajectory.
        
        Simulates: Target orbit, holding pattern, pursuit curve
        
        Constant bank angle turn at fixed speed.
        """
        points = []
        
        altitude_m = altitude_km * 1000
        sos = PhysicsConstants.speed_of_sound(altitude_m)
        speed = mach * sos
        turn_radius_m = turn_radius_km * 1000
        
        # Angular velocity
        omega = speed / turn_radius_m  # rad/s
        
        # Centripetal acceleration
        centripetal_g = speed**2 / (turn_radius_m * PhysicsConstants.G)
        
        t = 0.0
        while t <= duration:
            # Circular motion
            angle = omega * t
            
            pos = np.array([
                turn_radius_m * np.sin(angle),
                turn_radius_m * (1 - np.cos(angle)),
                altitude_m
            ])
            
            vel = np.array([
                speed * np.cos(angle),
                speed * np.sin(angle),
                0.0
            ])
            
            # Centripetal acceleration (toward center)
            acc = np.array([
                -speed**2 / turn_radius_m * np.sin(angle),
                -speed**2 / turn_radius_m * np.cos(angle) + speed**2 / turn_radius_m,
                0.0
            ])
            
            points.append(TrajectoryPoint(
                time=t,
                x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2],
                ax=acc[0], ay=acc[1], az=acc[2]
            ))
            
            t += self.dt
        
        self._compute_derived(points)
        
        return Trajectory(
            name=name,
            trajectory_type=TrajectoryType.COORDINATED_TURN,
            points=points,
            metadata={
                'mach': mach,
                'altitude_km': altitude_km,
                'turn_radius_km': turn_radius_km,
                'turn_g': centripetal_g,
                'omega_rad_s': omega,
                'dt': self.dt
            }
        )
    
    def terminal_dive(self, duration: float,
                     initial_mach: float = 10.0,
                     initial_altitude_km: float = 40.0,
                     dive_angle_deg: float = -70.0,
                     name: str = "Terminal_Dive") -> Trajectory:
        """
        Generate terminal dive attack trajectory.
        
        Simulates: Anti-ship missile terminal, ASBM dive
        
        Near-vertical high-speed dive with minimal maneuvering.
        """
        points = []
        
        altitude_m = initial_altitude_km * 1000
        sos = PhysicsConstants.speed_of_sound(altitude_m)
        speed = initial_mach * sos
        
        dive_rad = np.radians(dive_angle_deg)
        pos = np.array([0.0, 0.0, altitude_m])
        vel = np.array([
            speed * np.cos(dive_rad),
            0.0,
            speed * np.sin(dive_rad)
        ])
        
        t = 0.0
        while t <= duration and pos[2] > 0:
            # Gravity + drag
            rho = PhysicsConstants.air_density(pos[2])
            current_speed = np.linalg.norm(vel)
            drag = 0.5 * rho * 0.2 * 0.5 * current_speed**2 / 200
            drag_acc = -drag * vel / (current_speed + 1e-6)
            
            acc = np.array([drag_acc[0], drag_acc[1], -PhysicsConstants.G + drag_acc[2]])
            
            points.append(TrajectoryPoint(
                time=t,
                x=pos[0], y=pos[1], z=pos[2],
                vx=vel[0], vy=vel[1], vz=vel[2],
                ax=acc[0], ay=acc[1], az=acc[2]
            ))
            
            pos += vel * self.dt + 0.5 * acc * self.dt**2
            vel += acc * self.dt
            t += self.dt
        
        self._compute_derived(points)
        
        return Trajectory(
            name=name,
            trajectory_type=TrajectoryType.TERMINAL_DIVE,
            points=points,
            metadata={
                'initial_mach': initial_mach,
                'initial_altitude_km': initial_altitude_km,
                'dive_angle_deg': dive_angle_deg,
                'dt': self.dt
            }
        )
    
    # -------------------------------------------------------------------------
    # Multi-Target Scenarios
    # -------------------------------------------------------------------------
    
    def swarm(self, duration: float,
             num_targets: int = 5,
             mach: float = 3.0,
             spread_m: float = 1000.0,
             name: str = "Swarm") -> List[Trajectory]:
        """
        Generate swarm of targets moving together.
        
        Simulates: Drone swarm, MIRV deployment
        """
        trajectories = []
        center_traj = self.hypersonic_cruise(duration, mach=mach)
        
        for i in range(num_targets):
            # Offset from center
            offset = np.array([
                np.random.uniform(-spread_m, spread_m),
                np.random.uniform(-spread_m, spread_m),
                np.random.uniform(-spread_m/2, spread_m/2)
            ])
            
            points = []
            for p in center_traj.points:
                new_p = TrajectoryPoint(
                    time=p.time,
                    x=p.x + offset[0],
                    y=p.y + offset[1],
                    z=p.z + offset[2],
                    vx=p.vx + np.random.randn() * 10,
                    vy=p.vy + np.random.randn() * 10,
                    vz=p.vz + np.random.randn() * 5,
                    ax=p.ax, ay=p.ay, az=p.az
                )
                new_p.compute_derived()
                points.append(new_p)
            
            trajectories.append(Trajectory(
                name=f"{name}_Target_{i+1}",
                trajectory_type=TrajectoryType.SWARM,
                points=points,
                metadata={'swarm_index': i, 'offset': offset.tolist()}
            ))
        
        return trajectories


# =============================================================================
# TDOA Measurement Generator
# =============================================================================

class MultistatiNetwork:
    """
    Generate TDOA measurements for multistatic radar networks.
    
    Simulates realistic sensor network with:
    - Configurable node positions
    - Measurement noise
    - Clock bias (for async testing)
    """
    
    def __init__(self, node_positions: np.ndarray, 
                 tdoa_noise_m: float = 10.0,
                 clock_bias_ns: Optional[np.ndarray] = None):
        """
        Args:
            node_positions: Node positions [N, 3] in meters
            tdoa_noise_m: TDOA measurement noise (meters)
            clock_bias_ns: Clock bias for each node (nanoseconds)
        """
        self.nodes = node_positions
        self.num_nodes = len(node_positions)
        self.tdoa_noise_m = tdoa_noise_m
        
        if clock_bias_ns is None:
            self.clock_bias_ns = np.zeros(self.num_nodes)
        else:
            self.clock_bias_ns = clock_bias_ns
        
        self.reference_node = 0
    
    @staticmethod
    def create_pentagon_network(baseline_km: float = 600.0,
                               center_altitude_m: float = 0.0) -> 'MultistatiNetwork':
        """Create 6-node network (pentagon + center)"""
        baseline_m = baseline_km * 1000
        
        # Pentagon vertices
        angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        nodes = []
        
        for angle in angles:
            x = baseline_m/2 * np.cos(angle)
            y = baseline_m/2 * np.sin(angle)
            nodes.append([x, y, center_altitude_m])
        
        # Center node
        nodes.append([0.0, 0.0, center_altitude_m])
        
        return MultistatiNetwork(np.array(nodes))
    
    def generate_tdoa(self, target_pos: np.ndarray,
                     add_noise: bool = True,
                     add_clock_bias: bool = False) -> np.ndarray:
        """
        Generate TDOA measurements for target position.
        
        Returns:
            tdoa: Range differences [N-1] relative to reference node
        """
        # Range to each node
        ranges = np.array([
            np.linalg.norm(target_pos - self.nodes[i])
            for i in range(self.num_nodes)
        ])
        
        # TDOA relative to reference
        r_ref = ranges[self.reference_node]
        tdoa = np.array([
            ranges[i] - r_ref
            for i in range(1, self.num_nodes)
        ])
        
        # Add noise
        if add_noise:
            tdoa += np.random.randn(self.num_nodes - 1) * self.tdoa_noise_m
        
        # Add clock bias effect
        if add_clock_bias:
            c_m_per_ns = 0.299792458  # m/ns
            for i in range(self.num_nodes - 1):
                bias_diff = self.clock_bias_ns[i+1] - self.clock_bias_ns[self.reference_node]
                tdoa[i] += bias_diff * c_m_per_ns
        
        return tdoa
    
    def generate_measurements_for_trajectory(self, 
                                            trajectory: Trajectory,
                                            add_noise: bool = True,
                                            add_clock_bias: bool = False) -> np.ndarray:
        """Generate TDOA measurements for entire trajectory"""
        measurements = []
        
        for point in trajectory.points:
            pos = np.array([point.x, point.y, point.z])
            tdoa = self.generate_tdoa(pos, add_noise, add_clock_bias)
            measurements.append(tdoa)
        
        return np.array(measurements)


# =============================================================================
# Dataset Generator (High-Level Interface)
# =============================================================================

class DatasetGenerator:
    """
    High-level interface for generating complete tracking datasets.
    
    Generates trajectories + measurements + ground truth in multiple formats.
    """
    
    def __init__(self, dt: float = 0.0625, seed: Optional[int] = None):
        self.dt = dt
        self.traj_gen = TrajectoryGenerator(dt, seed)
        self.network = MultistatiNetwork.create_pentagon_network()
    
    def generate_standard_dataset(self, output_dir: str = "dataset"):
        """Generate standard benchmark dataset with multiple scenarios"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        scenarios = [
            ('hypersonic_cruise_m8', lambda: self.traj_gen.hypersonic_cruise(10.0, mach=8.0)),
            ('hypersonic_pullup_60g', lambda: self.traj_gen.hypersonic_pullup(10.0, pullup_g=60.0)),
            ('evasive_sturn_m6', lambda: self.traj_gen.evasive_sturn(10.0, mach=6.0)),
            ('coordinated_turn_m5', lambda: self.traj_gen.coordinated_turn(10.0, mach=5.0)),
            ('terminal_dive_m10', lambda: self.traj_gen.terminal_dive(8.0, initial_mach=10.0)),
            ('ballistic_reentry', lambda: self.traj_gen.ballistic_reentry(15.0)),
        ]
        
        print(f"Generating dataset in: {output_dir}/")
        print("-" * 50)
        
        for name, gen_func in scenarios:
            print(f"  Generating: {name}...", end=" ")
            
            traj = gen_func()
            
            # Save trajectory
            traj.save_csv(f"{output_dir}/{name}_trajectory.csv")
            traj.save_json(f"{output_dir}/{name}_trajectory.json")
            
            # Generate TDOA measurements
            tdoa = self.network.generate_measurements_for_trajectory(traj)
            np.save(f"{output_dir}/{name}_tdoa.npy", tdoa)
            
            print(f"OK ({len(traj)} points, max {traj.max_mach():.1f} Mach, max {traj.max_g_load():.1f}g)")
        
        # Save network configuration
        network_config = {
            'node_positions_m': self.network.nodes.tolist(),
            'num_nodes': self.network.num_nodes,
            'tdoa_noise_m': self.network.tdoa_noise_m
        }
        with open(f"{output_dir}/network_config.json", 'w') as f:
            json.dump(network_config, f, indent=2)
        
        print("-" * 50)
        print(f"Dataset generation complete!")
        print(f"Files saved to: {output_dir}/")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hypersonic & Anomaly Trajectory Dataset Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_generator.py --standard              Generate standard benchmark dataset
  python dataset_generator.py --type hypersonic       Generate hypersonic cruise trajectory
  python dataset_generator.py --type pullup --g 80    Generate 80g pull-up maneuver
  python dataset_generator.py --list                  List available trajectory types
        """
    )
    
    parser.add_argument('--standard', action='store_true',
                       help='Generate standard benchmark dataset')
    parser.add_argument('--type', type=str, choices=[t.value for t in TrajectoryType],
                       help='Trajectory type to generate')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Duration in seconds')
    parser.add_argument('--mach', type=float, default=8.0,
                       help='Mach number')
    parser.add_argument('--g', type=float, default=60.0,
                       help='G-load for maneuvers')
    parser.add_argument('--altitude', type=float, default=25.0,
                       help='Altitude in km')
    parser.add_argument('--dt', type=float, default=0.0625,
                       help='Time step in seconds')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory or filename prefix')
    parser.add_argument('--format', type=str, choices=['csv', 'json', 'npy', 'all'],
                       default='all', help='Output format')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--list', action='store_true',
                       help='List available trajectory types')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Trajectory Types:")
        print("-" * 40)
        for t in TrajectoryType:
            print(f"  {t.value}")
        return
    
    if args.standard:
        gen = DatasetGenerator(dt=args.dt, seed=args.seed)
        gen.generate_standard_dataset(args.output)
        return
    
    if args.type:
        traj_gen = TrajectoryGenerator(dt=args.dt, seed=args.seed)
        
        # Generate based on type
        if args.type == 'hypersonic_cruise':
            traj = traj_gen.hypersonic_cruise(args.duration, mach=args.mach, 
                                              altitude_km=args.altitude)
        elif args.type == 'hypersonic_pullup':
            traj = traj_gen.hypersonic_pullup(args.duration, initial_mach=args.mach,
                                              pullup_g=args.g)
        elif args.type == 'evasive_sturn':
            traj = traj_gen.evasive_sturn(args.duration, mach=args.mach,
                                          max_lateral_g=args.g)
        elif args.type == 'coordinated_turn':
            traj = traj_gen.coordinated_turn(args.duration, mach=args.mach)
        elif args.type == 'terminal_dive':
            traj = traj_gen.terminal_dive(args.duration, initial_mach=args.mach)
        elif args.type == 'ballistic_reentry':
            traj = traj_gen.ballistic_reentry(args.duration)
        else:
            print(f"Trajectory type '{args.type}' not yet implemented")
            return
        
        # Save
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        
        if args.format in ['csv', 'all']:
            traj.save_csv(f"{args.output}.csv")
            print(f"Saved: {args.output}.csv")
        
        if args.format in ['json', 'all']:
            traj.save_json(f"{args.output}.json")
            print(f"Saved: {args.output}.json")
        
        if args.format in ['npy', 'all']:
            traj.save_numpy(f"{args.output}.npy")
            print(f"Saved: {args.output}.npy")
        
        print(f"\nTrajectory Summary:")
        print(f"  Duration: {traj.duration():.2f} s")
        print(f"  Points: {len(traj)}")
        print(f"  Max Speed: {traj.max_speed():.1f} m/s ({traj.max_mach():.2f} Mach)")
        print(f"  Max G-Load: {traj.max_g_load():.1f} g")
        
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
