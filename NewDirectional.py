print("Starting Up")
from utils import *
import numpy as np
import glob
import time; end = lambda start: print(f"Done in {time.time()-start:.2f} seconds")
import sys; sys.stdout.flush()

# Point to Simulations
sims = np.sort(glob.glob('Halos/FoF/fiducial/*'))[500:550]

# Specify Snap (0:3, 1:2, 2:1, 3:0.5, 4:0)
snap = 3

# Specify number densities
Nrand = 10**6
Nmass = 10**5

# Specify kNNs
k = np.arange(128)+1
k = 2**np.arange(6)
k = np.array([1])

# Specify percentiles
p = np.logspace(-3, np.log10(0.5), 100)
p = np.append(p, 1-p[::-1][1:])*100

# Specify Stretches
S = [1.00, 1.02, 1.05, 0.98]

# Visualize 1e3 Data Points
viz = True

# Velocity Factor
vel_factor = 1.0

# 2D CDF
Ng = 50
low = 25
high = 30

# Specify Output Directory
if vel_factor == 1.0:
    result_dir = f'results_{snap}_newdir/'
else:
    result_dir = f'results_{snap}_newdir_{vel_factor:.1f}/'

import yt; yt.enable_parallelism()
#for q, sim in enumerate(sims):
for q, sim in yt.parallel_objects(enumerate(sims),0):

    # Specify exact sim path
    simid = int(os.path.basename(sim))
    sim_path = sim + f"/group_tab_{snap}/group_tab_{snap:03d}"
    print("checking", simid)

    # Check if other cpu is working on this sim
    check = False
    time.sleep(np.random.uniform(0,1))
    print("checking for",f"{result_dir}{simid:05d}.npz","in",f"{result_dir}+*")
    if f"{result_dir}{simid:05d}.npz" in glob.glob(f"{result_dir}*") and CanLoad(f"{result_dir}{simid:05d}.npz"):
        print(f"Skipping {simid:05d}")
        continue
    else:
        check = True
    os.system(f"touch {result_dir}{simid:05d}.npz")
    print(f"Working on {simid:05d}")
    start = time.time()

    # Load Data
    rpos, pos, _, rands, _, hubble, redshift, boxsize, cont = ReadData(sim, Nrand=Nrand, nmass = Nmass, snap=snap, vel_factor=vel_factor)

    pcdfs = []
    zcdfs = []
    zpcdfs = []
    zzcdfs = []
    r2cdfs = []
    z2cdfs = []
    As = []
    sys.stdout.flush()
    for s in S:
        if cont and check:

            # Stretch Data
            sdata, szdata, squery, sbs = stretch(rpos, pos, rands, boxsize, s)


            # Measure Real Space Quantities
            noRSD, noRSDids = kNNs(sdata, squery, K=k, bs=sbs, verb=False,getids=True)
            pp, rz, rd = PZD(sdata, squery, noRSD, noRSDids, avg=False,bs=sbs,verb=True, low=low, high=high)
            #pp, z, noRSDids = DistanceFilter(noRSD, low, high, pp, z, noRSDids)
            rN = len(rz)
            ra = minimize_loss(rd,rz)
            import pdb; pdb.set_trace()
            if int(simid) < 10 and s == 1.0 and viz:
                print("Lucky Winner: Visualizing")
                VisualizeSelection(sdata[np.random.choice(np.unique(noRSDids),1000, replace=False)].T, title=f"{result_dir}{simid}")
                print("Visualization Complete")
            print(f"2DCDF input: pp.shape = {pp[:,0].shape}; z.shape = {rz[:,0].shape}; high={high}, Ng={Ng}")
            r_2DCDF = TwoDimensionalCDF(pp[:,0], rz[:,0], high, Ng=Ng, verbose=False)[0]
            pp = CDF_percentile(pp, p)
            z = CDF_percentile(rz, p)

            # Measure z-Space Quantities
            RSD, RSDids = kNNs(szdata, squery, K=k, bs=sbs, verb=False, getids=True)
            zp, zz, zd = PZD(szdata, squery, RSD, RSDids, avg=False, bs=sbs, verb=True, low=low, high=high)
            #zp, zz = DistanceFilter(low, high, zp, zz)
            zN = len(zz)
            za = minimize_loss(zd,zz)
            print(ra.x, za.x, 1-za.x/ra.x)
            z_2DCDF = TwoDimensionalCDF(zp[:,0], zz[:,0], high, Ng=Ng, verbose=False)[0]
            zp = CDF_percentile(zp, p)
            zzz = CDF_percentile(zz, p)

            pcdfs.append(pp)
            zcdfs.append(z)
            zpcdfs.append(zp)
            zzcdfs.append(zzz)
            
            r2cdfs.append(r_2DCDF)
            z2cdfs.append(z_2DCDF)

            As.append([ra.x, za.x])

    if np.array(pcdfs).shape == (len(S), 2, len(p), len(k)) and np.array(zcdfs).shape == (len(S), 2, len(p), len(k)):

        pcdfs = np.array(pcdfs)
        zcdfs = np.array(zcdfs)
        zpcdfs = np.array(zpcdfs)
        zzcdfs = np.array(zzcdfs)

        r2cdfs = np.array(r2cdfs)
        z2cdfs = np.array(z2cdfs)

        N = np.array([rN, zN])
        As = np.array(As)

        try:
            print("Attempting to save to",f'{result_dir}{simid:05d}')
            np.savez(f'{result_dir}{simid:05d}',pcdfs=pcdfs, zcdfs=zcdfs, zpcdfs=zpcdfs, zzcdfs=zzcdfs, r2cdfs=r2cdfs, z2cdfs=z2cdfs, N=N, As=As)
        except:
            print('Save Failed!!')
    else:
        print(f"Misshapen Array for {simid:05d} =", np.array(pcdfs).shape)
        continue
    end(start)
    sys.stdout.flush()
