from utils import *
import numpy as np
import glob
import time; end = lambda start: print(f"Done in {time.time()-start:.2f} seconds")
import sys; sys.stdout.flush()

# Point to Simulations
sims = np.sort(glob.glob('Halos/FoF/fiducial/*'))

# Specify Snap (0:3, 1:2, 2:1, 3:0.5, 4:0)
#snaps = [2, 3]
#Zs = [1, 0.5]
snaps = [3]
Zs = [0.5]

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

# Specify Initial Point
Om0 = 0.3175
Om0grid = np.arange(0.25,0.35,0.1)
Om0grid = np.linspace(0.25,0.34,20)

# Velocity Factor
vel_factor = 1.0

# 2D CDF
Ng = 50

# Specify Output Directory

import yt; yt.enable_parallelism(); is_root = yt.is_root();

print(is_root)
#for q, sim in enumerate(sims):

def Loss(Om, Nmass, lo, hi):

    # Collects average As across different redshifts
    As = []
    zAs = []

    N = 0

    for z, snap in enumerate(snaps):

        # Collects measured As for every simulation at a given redshift
        A = []
        zA = []

        for q, sim in enumerate(sims[:]):


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
                    #A.append(getA(sdata, squery, sbs, lo, hi, False))
                    A.append(getKS(sdata, squery, sbs, lo, hi, False))

                    # Measure z-Space Quantities
                    #zA.append(getA(szdata, squery, sbs, lo, hi, False))
                    zA.append(getKS(szdata, squery, sbs, lo, hi, False))

                    N += 1

                    end = time.time()
                    if is_root:
                        print(f"Root process finished sim {q} in {end-start:.2f} seconds: Om = {Om:.2f}; a = {A[-1]:.5f}")

                except:
                    print("Couldn't get a")

            
            
        
        As.append(np.mean(A))
        zAs.append(np.mean(zA))
    #loss = np.array(As).std()
    #zloss = np.array(zAs).std()
    
    #print( "\n------------------------------------------------------")
    #print(f" Loss(Om={Om:.3f}) = { loss:.4f}; a = {np.mean( As):.4f};  da = { loss/np.sqrt(N):.4f}")
    #print(f"zLoss(Om={Om:.3f}) = {zloss:.4f}; a = {np.mean(zAs):.4f}; zda = {zloss/np.sqrt(N):.4f}")
    #print( "-------------------------------------------------------\n")

    return A, zA

from scipy.optimize import minimize

# Optimization
#start = time.time()
#Om0 = minimize(Loss, Om0, method='Nelder-Mead')
#end(start)
#print(Om)

params = [(0, 0.3175,   10**5, 0, 2),
          (1, 0.3175,   10**5, 2, 4),
          (2, 0.3175,   10**5, 4, 6),
          (3, 0.3175,   10**5, 6, 8),
          (4, 0.3175,   10**5, 8, 10),
          (5, 0.3175,   10**5, 10, 12),
          (6, 0.3175,   10**5, 12, 14),
          (7, 0.3175,   10**5, 14, 16),
          (8, 0.3175,   10**5, 16, 18),
          (9, 0.3175,   10**5, 18, 20),
          (10, 0.3175,   10**5, 20, 22),
          (11, 0.3175,   10**5, 22, 24),
          (12, 0.3175,   10**5, 24, 26),
          (13, 0.3175,   10**5, 26, 28),
          (14, 0.3175,   10**5, 28, 30),
          (15, 0.3175, 2*10**5, 28, 30),
          (16, 0.3175, 10**5//2, 28, 30),
          (17, 0.3175, 10**5//4, 28, 30),
          (18, 0.3175, 10**5//8, 28, 30)
          ]

'''
params = [(0, 0.3175,   10**5, 0, 100),
          (1, 0.3175,   10**5, 2, 100),
          (2, 0.3175,   10**5, 4, 100),
          (3, 0.3175,   10**5, 6, 100),
          (4, 0.3175,   10**5, 8, 100),
          (5, 0.3175,   10**5, 10, 100),
          (6, 0.3175,   10**5, 12, 100),
          (7, 0.3175,   10**5, 14, 100),
          (8, 0.3175,   10**5, 16, 100),
          (9, 0.3175,   10**5, 18, 100),
          (10, 0.3175,   10**5, 20, 100),
          (11, 0.3175,   10**5, 22, 100),
          (12, 0.3175,   10**5, 24, 100),
          (13, 0.3175,   10**5, 26, 100),
          (14, 0.3175,   10**5, 28, 100),
          (15, 0.3175, 2*10**5, 28, 30),
          (16, 0.3175, 10**5//2, 28, 30),
          (17, 0.3175, 10**5//4, 28, 30),
          (18, 0.3175, 10**5//8, 28, 30)
          ]
'''
labels = [f"{param[-2]}-{param[-1]}" for param in params[:15]]
#labels = [f">{param[-2]}" for param in params[:15]]
print(labels)


'''
losses = []
storage = {}
for sto, Om in yt.parallel_objects(Om0grid, 0, dynamic=False, storage=storage):

    sto.result = Loss(Om)
    sto.result_id = f"{Om:.3f}"
'''

storage = {}
for sto, param in yt.parallel_objects(params, 0, dynamic=False, storage=storage):

    id = param[0]

    sto.result = Loss(*param[1:])
    sto.result_id = f"{id}"


if yt.is_root():

    x = [float(key) for key in storage.keys()]
    print("ensure keys in order:", x)
    means = np.array([np.mean(result[0]) for result in storage.values()])
    zmeans = np.array([np.mean(result[1]) for result in storage.values()])
    stds = np.array([np.std(result[0])/np.sqrt(len(result[0])) for result in storage.values()])
    zstds = np.array([np.std(result[1])/np.sqrt(len(result[1])) for result in storage.values()])
    nzstds = np.array([np.std(result[1]) for result in storage.values()])

    ndense = np.array([-1, -2, -3, -5, -4])
    print(stds[ndense])

    plt.figure(figsize=(10,8))
    plt.errorbar([1.25e4, 2.5e4, 5e4, 1e5, 2e5], means[ndense], fmt='o', yerr=stds[ndense])
    plt.xscale('log')
    plt.xlabel(r"Number Density")
    plt.ylabel("a(z=0.5)")
    plt.title("Quijote Varying Number Density")
    plt.savefig("DirectionalParamScan_numberdensity_KS.png",dpi=230)


    ndist = np.arange(15)
    print(stds[ndist])

    plt.figure(figsize=(10,8))
    plt.errorbar(ndist, means[ndist], fmt='o', yerr=stds[ndist])
    plt.xticks(ndist, labels)
    plt.xlabel(r"Distance Scale Range")
    plt.ylabel("a(z=0.5)")
    plt.title("Quijote Varying Distance Scale")
    plt.savefig("DirectionalParamScan_scale_KS.png",dpi=230)

    print(zstds[ndense])

    plt.figure(figsize=(10,8))
    plt.errorbar([1.25e4, 2.5e4, 5e4, 1e5, 2e5], zmeans[ndense], fmt='o', yerr=zstds[ndense])
    plt.xscale('log')
    plt.xlabel(r"Number Density")
    plt.ylabel("a(z=0.5)")
    plt.title("Quijote Varying Number Density")
    plt.savefig("DirectionalParamScan_numberdensityz_KS.png",dpi=230)


    print(zstds[ndist])

    plt.figure(figsize=(10,8))
    plt.errorbar(ndist, zmeans[ndist], fmt='o', yerr=zstds[ndist])
    plt.xticks(ndist, labels)
    plt.xlabel(r"Distance Scale Range")
    plt.ylabel("a(z=0.5)")
    plt.title("Quijote Varying Distance Scale")
    plt.savefig("DirectionalParamScan_scalez_KS.png",dpi=230)

    sys.stdout.flush()


    # Save Results
    x = ndist
    y = zmeans[ndist]
    yerr = zstds[ndist]
    labels = labels
    np.savez('QPS_fiducial_KS',x=x,y=y,yerr=yerr,labels=labels, nzstds=nzstds[ndist])
