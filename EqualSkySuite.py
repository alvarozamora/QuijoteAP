import yt; yt.enable_parallelism(); is_root = yt.is_root();
from utils import *
import numpy as np
import glob
import time; end = lambda start: print(f"Done in {time.time()-start:.2f} seconds")
import sys; sys.stdout.flush()

import shelve

# Point to Simulations
sims = np.sort(glob.glob('Halos/FoF/fiducial/*')) # 15000
#sims = np.sort(glob.glob('Om_m/*'))   # 500
#sims = np.sort(glob.glob('Om_p/*'))   # 500

# Specify Snap (0:3, 1:2, 2:1, 3:0.5, 4:0)
#snaps = [2, 3]
#Zs = [1, 0.5]
snaps = [3]
Zs = [0.5]

# Specify number densities
Nrand = 10**6
Nmass = 10**5
subsamples = 10  # Best if some prime factor of Nmass

# Specify kNNs
k = np.arange(128)+1
k = 2**np.arange(6)
k = np.array([1,2,3,4])


# Specify Initial Point
Om0 = 0.3175
Om0grid = np.arange(0.25,0.35,0.1)
Om0grid = np.linspace(0.25,0.34,20)

# Velocity Factor
vel_factor = 1.0

# Specify Output Directory
#out = 'EqualSkySuite_Omm/'
#out = 'EqualSkySuite_Omp/'
#out = 'EqualSkySuite_OmpStretch/'
#out = 'EqualSkySuite_OmmStretch/'
out = 'EqualSkySuite/'
#out = 'EqualSkySuite_1.3/'
#out = 'EqualSkySuite_0.7/'
#out = 'EqualSkySuite_1.02/'
#out = 'EqualSkySuite_0.98/'

print(is_root)

for sim in yt.parallel_objects(sims, 0, dynamic=True):

    q = list(sims).index(sim)

    for z, snap in enumerate(snaps):

            try:
                with shelve.open(f'{out}{q:05d}') as db:
                    assert len(db['cCDF']) == 2, "Not Saved"
                    print(f"Skipping {q}")
                continue
            except:
                print(f"Starting {q}")


            start = time.time()

            # Specify exact sim path
            simid = int(os.path.basename(sim))
            sim_path = sim + f"/group_tab_{snap}/group_tab_{snap:03d}"

            # Load Data
            rpos, pos, _, rands, _, hubble, redshift, boxsize, cont = ReadData(sim, Nrand=Nrand, nmass = Nmass, snap=snap, vel_factor=vel_factor)

            sys.stdout.flush()
            s = sFactor(Om0 + 0.00, Zs[z])

            if cont:

                #try:

                sdata, szdata, squery, sbs = stretch(rpos, pos, rands, boxsize, s)

                cCDF  = compressedequalCDF( sdata, squery, sbs, k, False, subsamples)
                zcCDF = compressedequalCDF(szdata, squery, sbs, k, False, subsamples)


                end = time.time()
                if is_root:
                    print(f"Root process finished sim {q} in {end-start:.2f} seconds")

                with shelve.open(f'{out}{q:05d}') as db:
                    db['cCDF']  =  cCDF
                    db['zcCDF'] = zcCDF

                #except:
                #    print("Couldn't get a")
