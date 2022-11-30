'''
Creating a new model
====================

This example shows how to implement and use a new model.

'''

# %%
# Setup
# -----
# The setup here is the same as in the :doc:`/basic/plot_density` example,
# except that we use fewer galaxies and only a single matter shell.

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# import everything in the glass namespace
import glass.all
import glass

# also needs camb itself to get the parameter object
import camb

# decorator for GLASS generators
from glass.generator import generator

# import some variable names from other GLASS modules
# you do not have to do this, you can use their string values themselves
# but we generally do this to make life easier and prevent mistakes
from glass.matter import DELTA
from glass.galaxies import GAL_LON, GAL_LAT


# load the precomputed shell definition
shells, mweights, cls, lweights = glass.user.load_shells('shells.npz')

# basic parameters of the simulation
nside = 128
lmax = nside

# galaxy density
n_arcmin2 = 1e-6

# compute constant galaxy densities in shells
densities = glass.galaxies.constant_densities(n_arcmin2, shells)

# generators for a galaxies-only simulation with one correlated shell
generators = [
    glass.matter.gen_lognormal_matter(cls, nside, ncorr=1),
    glass.galaxies.gen_uniform_positions(densities),
]

# %%
# Model
# -----
# We will now add a custom model that returns a flag to indicate whether a
# galaxy is found in a matter overdensity or underdensity.
#
# The new model is prototypical of "inputs to outputs" models in GLASS.  Like
# all GLASS models, it is implemented as a simple Python generator.  The model
# can take global parameters, here a ``thresh`` value for the threshold at
# which matter is considered overdense.  The generator then runs in a loop: At
# each iteration of the simulation, it receives the matter density ``DELTA``
# and galaxy positions ``GAL_LON``, ``GAL_LAT``.  It then yields the newly
# computed overdensity flag ``GAL_OD_FLAG`` provided by the model.

# define a new variable for this example
# you could also just pass the string value, but we usually define these
GAL_OD_FLAG = 'galaxy overdensity flags'


# the decorator labels the inputs and outputs of this generator
# here we use the imported variable names from the GLASS modules
# but you could also provide their string values
@generator(
    receives=(DELTA, GAL_LON, GAL_LAT),
    yields=GAL_OD_FLAG)
def gal_od_flag_model(thresh=0.):

    # it's possible to pre-process before the iteration starts
    print('initialising our model')

    # this try .. except GeneratorExit statement contains the simulation loop
    try:
        # initial yield receives the first batch of values
        delta, lon, lat = yield

        # the main processing loop
        while True:
            # perform the computation for this iteration
            # get the HEALPix pixel index of the galaxies
            # set the flag if overdensity is above threshold
            nside = hp.get_nside(delta)
            ipix = hp.ang2pix(nside, lon, lat, lonlat=True)
            od = delta[ipix]
            od_flag = (od > thresh)

            # yield the result and receive new inputs
            delta, lon, lat = yield od_flag

    except GeneratorExit:
        # we are being told to stop processing
        print('stop processing inputs')

    # it's possible to post-process after the iteration stops
    print('finalising our model')


# add our new model to the generators used in the simulation
generators.append(gal_od_flag_model(thresh=0.01))


# %%
# Simulation
# ----------
# Run the simulation.  We will keep track of galaxy positions and their
# overdensity flags returned by our model.

# keep lists of positions and the overdensity flags
lon, lat, od_flag = np.empty(0), np.empty(0), np.empty(0, dtype=bool)

# outputs we want from the simulation
yields = [GAL_LON, GAL_LAT, GAL_OD_FLAG]

# simulate and add galaxies in each iteration to lists
for lon_i, lat_i, od_flag_i in glass.core.generate(generators, yields):
    lon = np.append(lon, lon_i)
    lat = np.append(lat, lat_i)
    od_flag = np.append(od_flag, od_flag_i)


# %%
# Visualisation
# -------------
# Show the positions of galaxies in underdense regions.

plt.subplot(111, projection='lambert')
plt.title('galaxies in underdensities')
plt.scatter(np.deg2rad(lon[~od_flag]), np.deg2rad(lat[~od_flag]), 8.0, 'r', alpha=0.5)
plt.grid(True)
plt.show()
