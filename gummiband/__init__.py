import numpy as np
from .curves import *
from .gummiband import Gummiband
from .topological_curves import ToroidalCurve
from .metric import NumericalMetric, Euclidean

from .gummiband import Gummiband, CollapsedToPointException
from .topological_curves import ToroidalCurve, ClosedToroidalCurve
from .schedule import create_more_points_on_convergence_schedule