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

metrics = []
def measure_metrics():

    # Define scales
    r = np.linspace(15, 20, 100)

    # Compute cdf residuals
    residuals = []
    #for q, CDF in enumerate(CDFs):
    for q, CDF in yt.parallel_objects(enumerate(CDFs), 0):
        residual = []

        if is_root:
            it = tqdm(range(N[q]))
        else:
            it = range(N[q])
        for n in it:

            if is_root:
                it.set_description(f"Processing {n+1}")

            # Extract LOS and perpendicular cdfs in redshift space
            cdf, zcdf = shelve.open(f"{CDF}{n:05d}").values()
            zlos, zprp = zcdf

            residual.append(np.abs(zlos[0].cdf(r) - zprp[0].cdf(r)).mean())

        residual = np.array(residual)
        residuals.append((residual.mean(), residual.std()))
    residuals = np.array(residuals)

    return residuals

a = residual_metric()
print(a)
    
    
