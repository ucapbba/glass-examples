'''
FITS file creation
=======================

This example demonstrates how to use the cls generated from shells.py to create a FITS catalog file

'''
from itertools import count
import numpy as np
from glass.galaxies import galaxy_shear
from glass.points import positions_from_delta
from glass.shapes import ellipticity_intnorm
from glass.lensing import from_convergence
from glass.user import swrite
import time

from createFITShelper import set_global_variables, set_glass_components

set_global_variables()
set_glass_components()

from createFITShelper import lmax, sigma_e, beff, binlabels, catalog, rng
from createFITShelper import matter, shells, ngal, convergence, fp, sum_n

start_time = time.time()
# open FITS file for writing
with swrite(catalog, ext="CATALOG") as out:
    # go through the shells and simulate
    for i, delta, shell in zip(count(), matter, shells):
        print("shell ", i)
        print("galaxy density: ", ngal[i])
        convergence.add_window(delta, shell)
        kappa = convergence.kappa
        gamma, = from_convergence(kappa, lmax, shear=True)
        galaxies = positions_from_delta(ngal[i], delta, beff, fp, rng=rng)
        for gal_lon, gal_lat, gal_count in galaxies:
            print("galaxies sampled: ", gal_count)
            # tomo bin id is predetermined by the population
            gal_b = np.repeat(binlabels, gal_count)
            gal_eps = ellipticity_intnorm(gal_count, sigma_e, rng=rng)
            gal_she = galaxy_shear(gal_lon, gal_lat, gal_eps,
                                   kappa, gamma.real, gamma.imag,
                                   reduced_shear=False)
            # some variance in shear weights
            gal_w = 10**rng.uniform(-2, 2, size=gal_she.shape)

            sum_n += gal_count

            out.write(
                RA=gal_lon,
                DEC=gal_lat,
                E1=gal_she.real,
                E2=gal_she.imag,
                W=gal_w,
                BIN=gal_b,
            )

totalTime = time.time()-start_time
print("--- glass FITS catalog creation took %s seconds ---" % totalTime)
