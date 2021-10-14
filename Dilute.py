print("Starting Up")
from utils import *
import numpy as np
import glob
import time; end = lambda start: print(f"Done in {time.time()-start:.2f} seconds")
import sys; sys.stdout.flush()

# Point to Simulations
sims = np.sort(glob.glob('Halos/FoF/fiducial/*'))

# Specify Snap (0:3, 1:2, 2:1)
snap = 2

# Specify number densities
Nrand = 10**6
Nmass = 10**4

# Specify kNNs
k = 2**np.arange(6)

# Specify percentiles
p = np.logspace(-3, np.log10(0.5), 100)
p = np.append(p, 1-p[::-1][1:])*100

# Specify Stretches
S = [1.00, 1.01, 1.02, 1.03, 1.05, 0.98]

# Specify Output Directory
result_dir = f'results_{snap}_dilute/'

for q, sim in enumerate(sims):

    # Specify exact sim path
    simid = int(os.path.basename(sim))
    sim_path = sim + f"/group_tab_{snap}/group_tab_{snap:03d}"
    print("checking", simid)

    # Check if other cpu is working on this sim
    check = False
    time.sleep(np.random.uniform(0,10))
    print("checking for",f"{result_dir}{simid:05d}.npz","in",f"{result_dir}+*")
    if f"{result_dir}{simid:05d}.npz" in glob.glob(f"{result_dir}*"):
        print(f"Skipping {simid:05d}")
        continue
    else:
        check = True
    os.system(f"touch {result_dir}{simid:05d}.npz")
    print(f"Working on {simid:05d}")
    start = time.time()

    # Load Data
    rpos, pos, _, rands, _, hubble, redshift, boxsize, cont = ReadData(sim, Nrand=Nrand, nmass = Nmass, snap=snap)

    cdfs = []
    zcdfs = []
    c2 = []
    zc2 = []
    sys.stdout.flush()
    for s in S:
        if cont and check:

            # Stretch Data
            sdata, szdata, squery, sbs = stretch(rpos, pos, rands, boxsize, s)


            # Measure Real Space Quantities
            noRSD = kNNs(sdata, squery, K=k, bs=sbs, verb=False)
            cdf = CDF_percentile(noRSD, p)

            # Measure z-Space Quantities
            RSD = kNNs(szdata, squery, K=k, bs=sbs, verb=False)
            zcdf = CDF_percentile(RSD, p)

            
            cdfs.append(cdf)
            zcdfs.append(zcdf)

    if np.array(cdfs).shape == (len(S), 2, len(p), len(k)) and np.array(zcdfs).shape == (len(S), 2, len(p), len(k)):
        cdfs = np.array(cdfs)
        zcdfs = np.array(zcdfs)

        print("Attempting to save to",f'{result_dir}{simid:05d}')
        np.savez(f'{result_dir}{simid:05d}',cdfs=cdfs,zcdfs=zcdfs)
    else:
        print(f"Misshapen Array for {simid:05d} =", np.array(cdfs).shape)
        continue
    end(start)
    sys.stdout.flush()