from utils import *
import numpy as np
import glob
import time; end = lambda start: print(f"Done in {time.time()-start:.2f} seconds")
import sys; sys.stdout.flush()

# Point to Simulations
sims = np.sort(glob.glob('Halos/FoF/fiducial/*'))

# Specify Snap (0:3, 1:2, 2:1, 3:0.5, 4:0)
snaps = [2, 3]
Zs = [1, 0.5]

# Specify number densities
Nrand = 10**7
Nmass = 10**5

# Specify kNNs
k = np.arange(128)+1
k = 2**np.arange(6)
k = np.array([1])

# Specify percentiles
p = np.logspace(-3, np.log10(0.5), 100)
p = np.append(p, 1-p[::-1][1:])*100

# Specify Initial Point
Om0 = 0.25
Om0grid = np.arange(0.25,0.35,0.1)
Om0grid = np.linspace(0.25,0.34,20)

# Velocity Factor
vel_factor = 1.0

# 2D CDF
Ng = 50
lo = 20
hi = 22

# Specify Output Directory

import yt; yt.enable_parallelism(); is_root = yt.is_root();

print(is_root)
#for q, sim in enumerate(sims):

def Loss(Om):

    # Collects average As across different redshifts
    As = []
    zAs = []

    N = 0

    for z, snap in enumerate(snaps):

        # Collects measured As for every simulation at a given redshift
        A = []
        zA = []

        for q, sim in enumerate(sims):


            start = time.time()

            # Specify exact sim path
            simid = int(os.path.basename(sim))
            sim_path = sim + f"/group_tab_{snap}/group_tab_{snap:03d}"

            # Load Data
            rpos, pos, _, rands, _, hubble, redshift, boxsize, cont = ReadData(sim, Nrand=Nrand, nmass = Nmass, snap=snap, vel_factor=vel_factor)

            sys.stdout.flush()
            s = sFactor(Om, Zs[z])

            if cont:

                try:

                    # Stretch Data
                    sdata, szdata, squery, sbs = stretch(rpos, pos, rands, boxsize, s)

                    # Measure Real Space Quantities
                    A.append(getA(sdata, squery, sbs, lo, hi, False))

                    # Measure z-Space Quantities
                    zA.append(getA(szdata, squery, sbs, lo, hi, False))

                    N += 1

                    end = time.time()
                    if is_root:
                        print(f"Root process finished sim {q} in {end-start:.2f} seconds: Om = {Om:.2f}; a = {A[-1]:.5f}")

                except:
                    print("Couldn't get a")

            
            
        
        As.append(np.mean(A))
        zAs.append(np.mean(zA))
    loss = np.array(As).std()
    zloss = np.array(zAs).std()
    
    print( "\n------------------------------------------------------")
    print(f" Loss(Om={Om:.3f}) = { loss:.4f}; a = {np.mean( As):.4f};  da = { loss/np.sqrt(N):.4f}")
    print(f"zLoss(Om={Om:.3f}) = {zloss:.4f}; a = {np.mean(zAs):.4f}; zda = {zloss/np.sqrt(N):.4f}")
    print( "-------------------------------------------------------\n")

    return loss, zloss

from scipy.optimize import minimize

# Optimization
#start = time.time()
#Om0 = minimize(Loss, Om0, method='Nelder-Mead')
#end(start)
#print(Om)

losses = []
storage = {}
for sto, Om in yt.parallel_objects(Om0grid, 0, dynamic=False, storage=storage):

    sto.result = Loss(Om)
    sto.result_id = f"{Om:.3f}"


if yt.is_root():
    x = [float(key) for key in storage.keys()]
    y = [result[0] for result in storage.values()]
    z = [result[1] for result in storage.values()]

    plt.figure(figsize=(10,8))
    plt.semilogy(x, y,'o')
    plt.semilogy([0.3175, 0.3175], [np.min(y), np.max(y)], 'k--')
    plt.xlabel(r"$\Omega_0$")
    plt.ylabel("Loss")
    plt.title("Quijote 1NN Loss Landscape")
    plt.savefig("DirectionalRedshift_aLoss.png",dpi=230)

    plt.figure(figsize=(10,8))
    plt.semilogy(x, z,'o')
    plt.semilogy([0.3175, 0.3175], [np.min(z), np.max(z)], 'k--')
    plt.xlabel(r"$\Omega_0$")
    plt.ylabel("Loss")
    plt.title("Quijote 1NN Loss Landscape")
    plt.savefig("DirectionalRedshift_azLoss.png",dpi=230)

    sys.stdout.flush()


    f = open('QuijoteDR_params','w')
    f.write(f'Nrand = {Nrand}\n')
    f.write(f'Nmass = {Nmass}\n')
    f.write(f'snaps = {snaps}\n')
    f.write(f'Zs    = {Zs}\n')
    f.write(f'lo    = {lo}\n')
    f.write(f'hi    = {hi}\n')
    f.close()

