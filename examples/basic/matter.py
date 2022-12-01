'''
Matter distribution
===================

This example simulates only the matter field in nested shells up to redshift 1.

'''

# %%
# Setup
# -----
# Set up a matter-only GLASS simulation, which only requires angulat matter
# power spectra and the sampling itself (here: lognormal).
#
# Uses the saved shell definitions from the :doc:`/basic/shells` example.

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# these are the GLASS imports: cosmology and everything in the glass namespace
import glass.all
import glass


# load the precomputed shell definition
shells, mweights, cls, lweights = glass.user.load_shells('shells.npz')

# basic parameters of the simulation
nside = 1024
zend = 1.

# generators for a matter-only simulation
# just the lognormal field with one correlated shell
generators = [
    glass.matter.gen_lognormal_matter(cls, nside, ncorr=1),
]


# %%
# Simulation
# ----------
# Run the simulation.  For each shell, plot an orthographic annulus of the
# matter distribution.

# make a 2d grid in redshift
n = 2000
zend = 1.05*shells[-1]
x, y = np.mgrid[-zend:zend:1j*n, -zend:zend:1j*n]
z = np.hypot(x, y)
grid = np.full(z.shape, np.nan)

# set up the plot
ax = plt.subplot(111)
ax.axis('off')

# simulate and project an annulus of each matter shell onto the grid
for i, delta_i in enumerate(glass.core.generate(generators, glass.matter.DELTA)):
    zmin, zmax = shells[i], shells[i+1]
    g = (zmin <= z) & (z < zmax)
    zg = np.sqrt(1 - (z[g]/zmax)**2)
    theta, phi = hp.vec2ang(np.transpose([x[g]/zmax, y[g]/zmax, zg]))
    grid[g] = hp.get_interp_val(delta_i, theta, phi)
    ax.add_patch(plt.Circle((0, 0), zmax/zend, fc='none', ec='k', lw=0.5, alpha=0.5, zorder=1))

# show the grid of shells
ax.imshow(grid, extent=[-1, 1, -1, 1], zorder=0,
          cmap='bwr', vmin=-2, vmax=2)

# show the resulting plot
plt.show()
