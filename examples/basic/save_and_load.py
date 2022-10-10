'''
Saving and loading
==================

This example demonstrates how generator outputs can be saved and loaded again.

This is mostly useful to accelerate repeated computations which use the same
input parameters.  For example, the angular matter power spectrum for a fixed
cosmology and matter weight function can be saved and then loaded again.  This
prevents re-running a costly computation many times over when different models
are being compared further down the simulation pipeline.

However, keep in mind that for large arrays, such as maps at high resolution, it
is almost always cheaper to generate the data again from a fixed seed and saved
inputs, than it is to save and load the large array itself.

'''

# %%
# Setup
# -----
# As the prototypical case of a generator that is expensive but only produces a
# relatively small amount of data, we will use CAMB to compute the angular
# matter power spectrum.

# these are the GLASS imports: cosmology and everything in the glass namespace
import glass.all
import glass

# also needs camb itself to get the parameter object
import camb


# cosmology for the simulation
h = 0.7
Oc = 0.25
Ob = 0.05

# set up CAMB parameters for matter angular power spectrum
pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2)

# basic parameters of the simulation
nside = 128
lmax = nside

# %%
# Saving
# ------
# Set up the costly ``camb_matter_cl`` computation, and then save the relevant
# outputs: matter shells, matter weight function, and angular matter power
# spectra.

# these are the variables that will be saved
save_vars = [
    glass.cosmology.ZMIN,
    glass.cosmology.ZMAX,
    glass.matter.WZ,
    glass.matter.CL,
]

# generators that are being saved
generators = [
    glass.cosmology.zspace(0., 1., num=5),
    glass.matter.mat_wht_redshift(),
    glass.camb.camb_matter_cl(pars, lmax),
    glass.core.save('my_saved_data.glass', save_vars),
]

# iterate; save data but here not doing anything else
print('saving ', end='')
for it in glass.core.generate(generators):
    print('.', end='')
print(' done!')

# %%
# Loading
# -------
# Now we can load the generator output from the saved file, and quickly run any
# number of generators.  For example, use the saved angular matter power spectra
# to generate a matter field.

# generators to load and process previously saved variables
generators = [
    glass.core.load('my_saved_data.glass'),
    glass.matter.lognormal_matter(nside),
]

# iterate; data is loaded and used by other generators
print('loading ', end='')
for it in glass.core.generate(generators):
    print('.', end='')
print(' done!')
