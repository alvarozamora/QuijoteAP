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

def residual(zlos, zprp, lo, hi):

    r = np.linspace(lo, hi, 100)

    return np.abs(zlos[0].cdf(r) - zprp[0].cdf(r)).mean()


limits = [
    (0,5),
    (5,10),
    (10,15),
    (15,20),
    (20,25),
    (25,30),
]

metrics = [residual]
def measure_metrics():


    # Compute cdf residuals
    all_measurements = []
    #for q, CDF in enumerate(CDFs):
    storage = {}
    for sto, (q, CDF) in yt.parallel_objects(enumerate(CDFs), 0, storage=storage):
        measurements = []

        if is_root:
            it = tqdm(range(N[q])[:1000])
        else:
            it = range(N[q])[:1000]
        for n in it:

            if is_root:
                it.set_description(f"Processing {n+1}")

            # Extract LOS and perpendicular cdfs in redshift space
            cdf, zcdf = shelve.open(f"{CDF}{n:05d}").values()
            zlos, zprp = zcdf

            dummy = []
            for (lo, hi) in limits:
                for metric in metrics:
                    dummy.append(metric(zlos, zprp, lo, hi))
            dummy = np.array(dummy)
            measurements.append(dummy)

        measurements = np.array(measurements).reshape(len(it),len(limits)) # Potential Bug Here
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
        plt.errorbar(x=range(len(a[CDF][0])), y=a[CDF][0], yerr=a[CDF][0]/np.sqrt(a[CDF][2]),fmt='.',label=legend_labels[q])
    plt.xlabel('Distance Scale (Mpc/h)')
    plt.ylabel('Absolute Residual')
    plt.xticks(range(len(labels)),labels)
    plt.legend()
    plt.savefig('testingsuite.png')



    
    
