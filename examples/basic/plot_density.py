'''
Galaxy distribution
===================

This example simulates a matter-only light cone up to redshift 1 and samples
galaxies from a uniform distribution in redshift.  The results are shown in a
pseudo-3D plot.  This helps to make sure the galaxies sampling across shells
works as intended.

'''

# %%
# Setup
# -----
# Set up a galaxy positions-only GLASS simulation.  It needs very little input:
# matter shell definition, a constant density distribution, and generators for
# uniform sampling of positions and redshifts.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# these are the GLASS imports: everything in the glass namespace
import glass.all
import glass


# create a basic setup of shells in redshift
shells = glass.matter.redshift_shells(0., 1., dz=0.1)

# total galaxy number density per unit redshift interval and arcmin2
dndz_arcmin2 = 0.01

# number density in each matter shell
ngal = glass.galaxies.constant_densities(dndz_arcmin2, shells)

# generators for a galaxies-only simulation with one correlated shell
generators = [
    glass.galaxies.gen_uniform_positions(ngal),
    glass.galaxies.gen_uniform_redshifts(shells),
]

# the values we want to get out of the simulation
yields = [
    glass.galaxies.GAL_Z,
    glass.galaxies.GAL_LON,
    glass.galaxies.GAL_LAT,
]


# %%
# Simulation
# ----------
# The goal of this example is to make a 3D cube of the sampled galaxy numbers.
# A redshift cube is initialised with zero counts, and the simulation is run.
# For every shell in the light cone, the galaxies are counted in the cube.

# make a cube for galaxy number in redshift
zbin = np.linspace(-shells[-1], shells[-1], 21)
cube = np.zeros((zbin.size-1,)*3)

# simulate and add galaxies in each matter shell to cube
for gal_z, gal_lon, gal_lat in glass.core.generate(generators, yields):
    z1 = gal_z*np.cos(np.deg2rad(gal_lon))*np.cos(np.deg2rad(gal_lat))
    z2 = gal_z*np.sin(np.deg2rad(gal_lon))*np.cos(np.deg2rad(gal_lat))
    z3 = gal_z*np.sin(np.deg2rad(gal_lat))
    (i, j, k), c = np.unique(np.searchsorted(zbin[1:], [z1, z2, z3]), axis=1, return_counts=True)
    cube[i, j, k] += c


# %%
# Visualisation
# -------------
# Lastly, make a pseudo-3D plot by stacking a number of density slices on top of
# each other.

# positions of grid cells of the cube
z = (zbin[:-1] + zbin[1:])/2
z1, z2, z3 = np.meshgrid(z, z, z)

# plot the galaxy distribution in pseudo-3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
norm = LogNorm(vmin=np.min(cube[cube>0]), vmax=np.max(cube), clip=True)
for i in range(len(zbin)-1):
    v = norm(cube[..., i])
    c = plt.cm.inferno(v)
    c[..., -1] = 0.2*v
    ax.plot_surface(z1[..., i], z2[..., i], z3[..., i], rstride=1, cstride=1,
                    facecolors=c, linewidth=0, shade=False, antialiased=False)
fig.tight_layout()
plt.show()
