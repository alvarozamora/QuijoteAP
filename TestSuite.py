import numpy as np
import matplotlib.pyplot as plt
import shelve
from tqdm import tqdm
import yt; yt.enable_parallelism(); is_root = yt.is_root();


CDFs = [
    'DirectionalSuite/', 
    'DirectionalSuite_OmpStretch/', 
    'DirectionalSuite_OmmStretch/', 
    'DirectionalSuite_Omp/', 
    'DirectionalSuite_Omm/', 
]

N = [
    15000, 
    15000, 
    15000, 
    500, 
    500
]
Nboxes = 15000

def residual(zlos, zprp, lo, hi):

    r = np.linspace(lo, hi, 100)

    return np.abs(zlos[0].cdf(r) - zprp[0].cdf(r)).mean()

nr = 50
def k_residual(zlos, zprp, k=0):

    r = np.logspace(np.log10(1), np.log10(30), nr)

    return zlos[k].cdf(r) - zprp[k].cdf(r)



limits = [
    (0,5),
    (5,10),
    (10,15),
    (15,20),
    (20,25),
    (25,30),
]
'''
limits = [
    (0,100),
    (5,100),
    (10,100),
    (15,100),
    (20,100),
    (25,100),
]
'''

metrics = [k_residual]
def measure_metrics():


    # Compute cdf residuals
    all_measurements = []
    #for q, CDF in enumerate(CDFs):
    storage = {}
    for sto, (q, CDF) in yt.parallel_objects(enumerate(CDFs), 0, storage=storage):
        measurements = []

        if is_root:
            it = tqdm(range(N[q])[:Nboxes])
        else:
            it = range(N[q])[:Nboxes]
        for n in it:

            if is_root:
                it.set_description(f"Processing {n+1}")

            # Extract LOS and perpendicular cdfs in redshift space
            cdf, zcdf = shelve.open(f"{CDF}{n:05d}").values()
            zlos, zprp = zcdf

            for metric in metrics:
                measurements.append(metric(zlos, zprp, k=0))

        measurements = np.array(measurements).reshape(len(it), nr) # Potential Bug Here
        sto.result = (measurements.mean(axis=0), measurements.std(axis=0), measurements.shape[0])
        sto.result_id = CDF

    return storage

a = measure_metrics()
#print(a[CDFs[0]])

if is_root:
    labels = [f"{limit[0]}-{limit[1]}" for limit in limits]
    legend_labels = ['fid', r'fid with $\Omega_+$ stretch', r'fid with $\Omega_-$ stretch', r'$\Omega_+$',r'$\Omega_-$']
    plt.figure()
    for q, CDF in enumerate(CDFs):
        plt.errorbar(x=range(len(a[CDF][0])), y=a[CDF][0], yerr=a[CDF][1]/np.sqrt(a[CDF][2]),fmt='.',label=legend_labels[q])
    plt.xlabel('Distance Scale (Mpc/h)')
    plt.ylabel('Absolute Residual')
    plt.xticks(range(len(labels)),labels)
    plt.legend()
    plt.savefig('testingsuite.png')

if is_root:
    legend_labels = ['fid', r'fid with $\Omega_+$ stretch', r'fid with $\Omega_-$ stretch', r'$\Omega_+$',r'$\Omega_-$']
    plt.figure()
    r = np.logspace(np.log10(1), np.log10(30), nr)
    for q, CDF in enumerate(CDFs[3:], 3):
        plt.errorbar(x=r, y=a[CDF][0]-a[CDFs[0]][0], yerr=a[CDF][1]/np.sqrt(a[CDF][2]),fmt='.',label=legend_labels[q])
    for q, CDF in enumerate(CDFs[1:3], 1):
        plt.errorbar(x=r, y=a[CDF][0]-a[CDFs[0]][0], yerr=a[CDF][1]/np.sqrt(a[CDF][2]),fmt='.',label=legend_labels[q])
    plt.gca().set_xscale('log')
    plt.xlabel('Distance Scale (Mpc/h)')
    plt.ylabel('Absolute Residual')
    plt.legend()
    plt.savefig('testingsuite.png')





    
    
