import numpy as np
import glob
import time; end = lambda start: print(f"Done in {time.time()-start:.2f} seconds")
import sys; sys.stdout.flush()
import matplotlib.pyplot as plt


fiducial_measurements = np.load('QPS_fiducial_100_1e7.npz')
Omp =  np.load('QPS_Omp_100_1e7.npz')
Omm =  np.load('QPS_Omm_100_1e7.npz')
Omps =  np.load('QPS_fiducial_100_OmpStretch.npz')
Omms =  np.load('QPS_fiducial_100_OmmStretch.npz')

ndist = np.arange(15)
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

labels = [f">{param[-2]}" for param in params[:15]]

plt.figure(figsize=(10,8))
plt.errorbar(ndist, Omm['y'], fmt='o', yerr=Omm['yerr'],label=r'$\Omega_m^-$')
plt.errorbar(ndist, Omp['y'], fmt='o', yerr=Omp['yerr'],label=r'$\Omega_m^+$')
plt.errorbar(ndist, fiducial_measurements['y'], fmt='o', yerr=fiducial_measurements['yerr'],label='fid')
plt.errorbar(ndist, Omms['y'], fmt='o', yerr=Omms['yerr'],label=r'fid+$\Omega_m^-$ stretch')
plt.errorbar(ndist, Omps['y'], fmt='o', yerr=Omps['yerr'], label=r'fid+$\Omega_m^+$ stretch')
plt.xticks(ndist, labels)
plt.xlabel(r"Distance Scale Range")
plt.ylabel("a(z=0.5)")
plt.title("Quijote Varying Distance Scale")
plt.legend()
plt.savefig(f"DirectionalParamScan_scalez_100_all.png",dpi=230)