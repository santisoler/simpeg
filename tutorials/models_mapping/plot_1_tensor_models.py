

"""
Tensor Meshes
=============

Here we demonstrate various ways that models can be defined and mapped to
tensor meshes. Some things we consider are:

    - Surface topography
    - Adding structures of various shape to the model
    - Parameterized models
    - Models with 2 or more physical properties
    

"""

#########################################################################
# Import modules
# --------------
#

from discretize import TensorMesh
from SimPEG.Utils import mkvc, surface2ind_topo, ModelBuilder
from SimPEG import Maps
import numpy as np
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 3

#############################################
# Defining the mesh
# -----------------
#
# Here, we create the tensor mesh that will be used for all examples.
#


def make_example_mesh():

    dh = 5.
    hx = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
    hy = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
    hz = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
    mesh = TensorMesh([hx, hy, hz], 'CCC')

    return mesh


#############################################
# Halfspace model with topography at z = 0
# ----------------------------------------
#
# In this example we generate a half-space model. Since air cells remain
# constant during geophysical inversion, the number of model values we define
# should be equal to the number of cells lying below the surface. Here, we
# define the model (*mod* ) as well as the mapping (*mod_map* ) that goes from
# the model-space to the entire mesh.
#

mesh = make_example_mesh()

halfspace_val = 100.

# Find cells below topography and define mapping
air_val = 0.
ind_active = mesh.gridCC[:, 2] < 0.
mod_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define the model
mod = halfspace_val*np.ones(ind_active.sum())

# We can plot a slice of the model at Y=-2.5
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plotSlice(mod_map*mod, normal='Y', ax=ax, ind=int(mesh.nCy/2), grid=True)
ax.set_title('Model slice at y = -2.5 m')
plt.show()

#############################################
# Topography, a block and a vertical dyke
# ---------------------------------------
#
# In this example we create a model containing a block and a vertical dyke
# that strikes along the y direction. The utility *surface2ind_topo* is used
# to find the cells which lie below a set of xyz points defining a surface.
#

mesh = make_example_mesh()

background_val = 100.
dyke_val = 40.
block_val = 70.

# Define surface topography as an (N, 3) np.array. You could also load a file
# containing the xyz points
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 75**2) + 40.
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# Find cells below topography and define mapping
air_val = 0.
ind_active = surface2ind_topo(mesh, topo, 'N')
mod_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define the model on subsurface cells
mod = background_val*np.ones(ind_active.sum())
ind_dyke = (mesh.gridCC[ind_active, 0] > 20.) & (mesh.gridCC[ind_active, 0] < 40.)
mod[ind_dyke] = dyke_val
ind_block = (
    (mesh.gridCC[ind_active, 0] > -40.) & (mesh.gridCC[ind_active, 0] < -10.) &
    (mesh.gridCC[ind_active, 1] > -30.) & (mesh.gridCC[ind_active, 1] < 30.) &
    (mesh.gridCC[ind_active, 2] > -40.) & (mesh.gridCC[ind_active, 2] < 0.)
)
mod[ind_block] = block_val

# Plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plotSlice(mod_map*mod, normal='Y', ax=ax, ind=int(mesh.nCy/2), grid=True)
ax.set_title('Model slice at y = -2.5 m')
plt.show()


#############################################
# Combo Maps
# ----------
#
# Here we demonstrate how combo maps can be used to create a single mapping
# from the model to the mesh. In this case, our model consists of
# log-conductivity values but we want to plot the resistivity. To accomplish
# this we must take the exponent of our model values, then take the reciprocal,
# then map from below surface cell to the mesh.
#

mesh = make_example_mesh()

background_val = np.log(1./100.)
dyke_val = np.log(1./40.)
block_val = np.log(1./70.)

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 75**2) + 40.
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# Find cells below topography
air_val = 0.
ind_active = surface2ind_topo(mesh, topo, 'N')
active_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define the model on subsurface cells
mod = background_val*np.ones(ind_active.sum())
ind_dyke = (mesh.gridCC[ind_active, 0] > 20.) & (mesh.gridCC[ind_active, 0] < 40.)
mod[ind_dyke] = dyke_val
ind_block = (
    (mesh.gridCC[ind_active, 0] > -40.) & (mesh.gridCC[ind_active, 0] < -10.) &
    (mesh.gridCC[ind_active, 1] > -30.) & (mesh.gridCC[ind_active, 1] < 30.) &
    (mesh.gridCC[ind_active, 2] > -40.) & (mesh.gridCC[ind_active, 2] < 0.)
)
mod[ind_block] = block_val

# Define a single mapping from model to mesh
exp_map = Maps.ExpMap()
rec_map = Maps.ReciprocalMap()
mod_map = Maps.ComboMap([active_map, rec_map, exp_map])

# Plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plotSlice(mod_map*mod, normal='Y', ax=ax, ind=int(mesh.nCy/2), grid=True)
ax.set_title('Model slice at y = -2.5 m')
plt.show()


#############################################
# Models with arbitrary shapes
# ----------------------------
#
# Here we show how model building utilities are used to make more complicated
# structural models. The process of adding a new unit is twofold: 1) we must
# find the indicies for mesh cells that lie within the new unit, 2) we
# replace the prexisting physical property value for those cells.
#

mesh = make_example_mesh()

background_val = 100.
dyke_val = 40.
sphere_val = 70.

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 75**2) + 40.
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# Set active cells and define unit values
air_val = 0.
ind_active = surface2ind_topo(mesh, topo, 'N')
mod_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define model for cells under the surface topography
mod = background_val*np.ones(ind_active.sum())

# Add a sphere
ind_sphere = ModelBuilder.getIndicesSphere(
    np.r_[-25., 0., -15.], 20., mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]  # So it's same size and order as model
mod[ind_sphere] = sphere_val

# Add dyke defined by a set of points
xp = np.kron(np.ones((2)), [-10., 10., 45., 25.])
yp = np.kron([-1000., 1000.], np.ones((4)))
zp = np.kron(np.ones((2)), [-120., -120., 35., 35.])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind_poly = ModelBuilder.PolygonInd(mesh, xyz_pts)
ind_poly = ind_poly[ind_active]  # So same size and order as model
mod[ind_poly] = dyke_val

# Plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plotSlice(mod_map*mod, normal='Y', ax=ax, ind=int(mesh.nCy/2), grid=True)
ax.set_title('Model slice at y = -2.5 m')
plt.show()


#############################################
# Parameterized block model
# -------------------------
#
# Instead of defining a model value for each sub-surface cell, we can define
# the model in terms of a small number of parameters. Here we parameterize the
# model as a block in a half-space. We then create a mapping which projects
# this model onto the mesh.
#

mesh = make_example_mesh()

background_val = 100.        # background value
block_val = 40.              # block value
xc, yc, zc = -25., 0., -20.  # center of block
dx, dy, dz = 30., 40., 30.   # dimensions in x,y,z

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 75**2) + 40.
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# Set active cells and define unit values
air_val = 0.
ind_active = surface2ind_topo(mesh, topo, 'N')
active_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define the model on subsurface cells
mod = np.r_[background_val, block_val, xc, dx, yc, dy, zc, dz]
param_map = Maps.ParametricBlock(mesh, indActive=ind_active, epsilon=1e-10, p=5.)

# Define a single mapping from model to mesh
mod_map = Maps.ComboMap([active_map, param_map])

# Plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plotSlice(mod_map*mod, normal='Y', ax=ax, ind=int(mesh.nCy/2), grid=True)
ax.set_title('Model slice at y = -2.5 m')
plt.show()


#############################################
# Using Wire Maps
# ---------------
#
# Wire maps are needed when the model is comprised of two or more parameter
# types (e.g. conductivity and magnetic permeability). Because the model
# vector contains all values for all parameter types, we need to use "wires"
# to extract the values for a particular parameter type.
#
# Here we will define a model consisting of log-conductivity values and
# magnetic permeability values. We wish to plot the conductivity and
# permeability on the mesh. Wires are used to keep track of the mapping
# between the model vector and a particular physical property type.
#

mesh = make_example_mesh()

sig_back = np.log(100.)
sig_sph = np.log(70.)
sig_dyke = np.log(40.)
mu_back = 1.
mu_sph = 1.25

# Define surface topography
[xx, yy] = np.meshgrid(mesh.vectorNx, mesh.vectorNy)
zz = -3*np.exp((xx**2 + yy**2) / 75**2) + 40.
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# Set active cells
air_val = 0.
ind_active = surface2ind_topo(mesh, topo, 'N')
active_map = Maps.InjectActiveCells(mesh, ind_active, air_val)

# Define model for cells under the surface topography
N = int(ind_active.sum())
mod = np.kron(np.ones((N, 1)), np.c_[sig_back, mu_back])

# Add a conductive and permeable sphere
ind_sphere = ModelBuilder.getIndicesSphere(
    np.r_[-25., 0., -15.], 20., mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]  # So same size and order as model
mod[ind_sphere, :] = np.c_[sig_sph, mu_sph]

# Add a conductive and non-permeable dyke
xp = np.kron(np.ones((2)), [-10., 10., 45., 25.])
yp = np.kron([-1000., 1000.], np.ones((4)))
zp = np.kron(np.ones((2)), [-120., -120., 35., 35.])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind_poly = ModelBuilder.PolygonInd(mesh, xyz_pts)
ind_poly = ind_poly[ind_active]  # So same size and order as model
mod[ind_poly, 0] = sig_dyke

# Create model vector and wires
mod = mkvc(mod)
wire_map = Maps.Wires(('logsig', N), ('mu', N))

# Use combo maps to map from model to mesh
sig_map = Maps.ComboMap([active_map, Maps.ExpMap(), wire_map.logsig])
mu_map = Maps.ComboMap([active_map, wire_map.mu])

# Plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plotSlice(sig_map*mod, normal='Y', ax=ax, ind=int(mesh.nCy/2), grid=True)
ax.set_title('Model slice at y = -2.5 m')
plt.show()
