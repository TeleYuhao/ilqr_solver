"""ALM iLQR trajectory optimization library.

This package implements an Augmented Lagrangian Method (ALM) combined with
the Iterative Linear Quadratic Regulator (iLQR) for constrained trajectory
optimization of autonomous vehicles.

This is a self-contained package with no external dependencies on scripts_new.

Main components:
    - ModelBase: Abstract base class for kinematic models
    - Obstacle: Obstacle class with ellipsoidal safety margins
    - ALMILQRCore: Two-loop ALM iLQR solver
    - ALMModel: Vehicle kinematic model with ALM constraints
"""

from alm_ilqr_core import ALMILQRCore
from alm_model import ALMModel
from model_base import ModelBase
from obstacle import Obstacle

__all__ = ['ModelBase', 'Obstacle', 'ALMILQRCore', 'ALMModel']
__version__ = '3.0.0'
