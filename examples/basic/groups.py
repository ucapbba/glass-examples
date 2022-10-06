'''
Generator groups
================

This example demonstrates how generators can be grouped.  Groups provide a
layered namespace for outputs.  This is useful if similar generators are run
for different populations of objects, since it removes the need to manually
change input and output names.

'''

# %%
# Setup
# -----
# The simplest galaxies-only GLASS simulation, sampling galaxies uniformly over
# the sphere using some redshift distribution.  Galaxies are sampled in two
# groups: low and high redshifts.

import numpy as np
import matplotlib.pyplot as plt

# import everything in the glass namespace
import glass.all
import glass


# basic parameters of the simulation
nside = 128
lmax = nside

# galaxy density
n_arcmin2 = 1e-4

# parametric galaxy redshift distribution
z = np.linspace(0, 3, 301)
dndz_low = n_arcmin2*glass.observations.smail_nz(z, 0.5, 1.0, 2.5)
dndz_high = n_arcmin2*glass.observations.smail_nz(z, 2.0, 4.0, 2.5)

# generators for a uniform galaxies simulation
generators = [
    glass.cosmology.zspace(0., 3., dz=0.25),
    glass.core.group('low-z', [
        glass.galaxies.gal_density_dndz(z, dndz_low),
        glass.galaxies.gal_positions_unif(),
        glass.galaxies.gal_redshifts_nz(),
    ]),
    glass.core.group('high-z', [
        glass.galaxies.gal_density_dndz(z, dndz_high),
        glass.galaxies.gal_positions_unif(),
        glass.galaxies.gal_redshifts_nz(),
    ]),
]


# %%
# Simulation
# ----------
# Keep the simulated redshifts of both populations.  Note how the groups provide
# a nested namespace for the data.

# arrays for true (ztrue) and photmetric (zphot) redshifts
low_z = np.empty(0)
high_z = np.empty(0)

# simulate and add galaxies in each matter shell to arrays
for shell in glass.core.generate(generators):
    low_z = np.append(low_z, shell['low-z'][glass.galaxies.GAL_Z])
    high_z = np.append(high_z, shell['high-z'][glass.galaxies.GAL_Z])


# %%
# Plots
# -----
# Plot the two distributions together with the expected inputs.

norm = glass.util.ARCMIN2_SPHERE*(z[-1] - z[0])/40

for zz, nz, label in (low_z, dndz_low, 'low-z'), (high_z, dndz_high, 'high-z'):
    plt.hist(zz, bins=40, range=(z[0], z[-1]), histtype='stepfilled', alpha=0.5, label=label)
    plt.plot(z, norm*nz, '-k', lw=1, alpha=0.5)
plt.xlabel('redshift $z$')
plt.ylabel('number of galaxies')
plt.legend()
plt.show()
