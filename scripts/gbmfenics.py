#%%
from dolfin import *
import numpy as np
from matplotlib import pyplot as plt


mesh = SphericalShellMesh.create(MPI.comm_world, 1)


n = SpatialCoordinate(mesh)