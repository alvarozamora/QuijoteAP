import shelve

from matplotlib.pyplot import tight_layout
import numpy as np
from utils import *

# Take database off shelve
#db = shelve.open('quijote_cdfs_sigmas')

# Extract results from database
#results = db['results']
#CDFs, cumulants, zCDFs, zcumulants = extract_results()#results)


#import pdb; pdb.set_trace()
#avg = CDFs.mean(0)
#zavg = zCDFs.mean(0)
avg, zavg = extract_results()
import pdb; pdb.set_trace()
print(len(avg), len(zavg))
stdev =  avg.std(0)/np.sqrt(len(avg))
ztdev = zavg.std(0)/np.sqrt(len(zavg))
avg  =  avg.mean(0)
zavg = zavg.mean(0)

S = [1, 1.01, 1.02, 1.03, 1.05, 0.98]
k = [1, 2, 3, 4, 8]
k = np.arange(32)+1
k = 2**np.arange(8)-1
k = 2**np.arange(13)


print("Plotting CDFs")
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        cdf = avg[q,1,:, kk]
        ax[kk].loglog(avg[q,0,:, kk], cdf, '.-', label=f'{s:.2f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF')
    ax[kk].set_title(f'{knn}NN Peaked CDF')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
    ax[kk].set_ylim(1e-3,1)
plt.tight_layout()
plt.savefig('cdf_results.png')

print("Plotting CDF Ratios")
fig, ax = plt.subplots(len(k), 1, figsize=(10,5*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S[1:], 1):
        #ratio = avg[q,0,:, kk]/avg[0,0,:, kk]
        #ax[kk].semilogx(avg[q,0,:, kk], ratio, '.-', label=f'{s:.2f}')

        ratio = avg[q,0,:, kk]/avg[0,0,:, kk]
        error = np.sqrt( (stdev[q,0,:,kk]/avg[0,0,:, kk])**2 + (stdev[q,0,:,kk] * avg[q,0,:, kk]/avg[0,0,:, kk]**2)**2)

        ax[kk].errorbar(avg[q,0,:, kk], ratio, yerr=error, label=f'{s:.2f}')
        ax[kk].set_xscale('log')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN s/u Distance Ratio')
    ax[kk].set_title(f'{knn}NN s/u Distance Ratio')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('cdf_ratio_results.png')
'''
print("Plotting Second Cumulants")
fig, ax = plt.subplots(1, 1, figsize=(6,8))
for q, s in enumerate(S):
    r, C2 = cumulants.mean(0)[q]
    ax.loglog(r, C2, '.-', label=f'{s:.2f}')
analytic = second_cumulant_analytic('planck18', r, 1e5/(1e3)**3)
ax.semilogx(r, analytic, label='colossus')
ax.set_xlabel(r'Distance (Mpc $h^{-1}$)')
ax.set_ylabel(f'Second Cumulant')
ax.set_title(f'Second Cumulant')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('second_cumulant.png')

print("Plotting Second Cumulant Ratios")
fig, ax = plt.subplots(1, 1, figsize=(6,8))
for q, s in enumerate(S[1:],1):
    r, C2 = cumulants.mean(0)[q]
    ratio = C2/cumulants.mean(0)[0][1]
    ax.semilogx(r, ratio, '.-', label=f'{s:.2f}')
ax.set_xlabel(r'Distance (Mpc $h^{-1}$)')
ax.set_ylabel(f'Second Cumulant Ratio')
ax.set_title(f'Second Cumulant Ratio')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('second_cumulant_ratio.png')
'''


## REDSHIFT PLOTS

print("Plotting zCDFs")
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        cdf = zavg[q,1,:, kk]
        ax[kk].loglog(zavg[q,0,:, kk], cdf, '.-', label=f'{s:.2f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF')
    ax[kk].set_title(f'{knn}NN Peaked CDF')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
    ax[kk].set_ylim(1e-2,1)
plt.tight_layout()
plt.savefig('zcdf_results.png')

print("Plotting zCDF Ratios")
fig, ax = plt.subplots(len(k), 1, figsize=(10,5*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S[1:], 1):
        ratio = zavg[q,0,:, kk]/zavg[0,0,:, kk]
        error = np.sqrt( (ztdev[q,0,:,kk]/zavg[0,0,:, kk])**2 + (ztdev[q,0,:,kk] * zavg[q,0,:, kk]/zavg[0,0,:, kk]**2)**2)

        ax[kk].errorbar(zavg[q,0,:, kk], ratio, yerr=error, label=f'{s:.2f}')
        ax[kk].set_xscale('log')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN s/u Distance Ratio')
    ax[kk].set_title(f'{knn}NN s/u Distance Ratio')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('zcdf_ratio_results.png')

'''
print("Plotting Second zCumulants")
fig, ax = plt.subplots(1, 1, figsize=(6,8))
for q, s in enumerate(S):
    r, C2 = zcumulants.mean(0)[q]
    ax.loglog(r, C2, '.-', label=f'{s:.2f}')
analytic = second_cumulant_analytic('planck18', r, 1e5/(1e3)**3)
ax.semilogx(r, analytic, label='colossus')
ax.set_xlabel(r'Distance (Mpc $h^{-1}$)')
ax.set_ylabel(f'Second Cumulant')
ax.set_title(f'Second Cumulant')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('second_zcumulant.png')

print("Plotting Second zCumulant Ratios")
fig, ax = plt.subplots(1, 1, figsize=(6,8))
for q, s in enumerate(S[1:],1):
    r, C2 = zcumulants.mean(0)[q]
    ratio = C2/zcumulants.mean(0)[0][1]
    ax.semilogx(r, ratio, '.-', label=f'{s:.2f}')
ax.set_xlabel(r'Distance (Mpc $h^{-1}$)')
ax.set_ylabel(f'Second Cumulant Ratio')
ax.set_title(f'Second Cumulant Ratio')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('second_zcumulant_ratio.png')
'''


## COMPARISONS


print("Plotting both CDFs")
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        cdf = avg[q,1,:, kk]
        ax[kk].loglog(avg[q,0,:, kk], cdf, f'C{q}', label=f'{s:.2f}')

        zcdf = zavg[q,1,:, kk]
        ax[kk].loglog(zavg[q,0,:, kk], zcdf, f'C{q}--', label=f'{s:.2f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF')
    ax[kk].set_title(f'{knn}NN Peaked CDF')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
    ax[kk].set_ylim(1e-3,1)
plt.tight_layout()
plt.savefig('bothcdf_results.png')

print("Plotting both Ratios")
fig, ax = plt.subplots(len(k), 1, figsize=(10,5*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):

        ratio = zavg[q,0,:, kk]/avg[q,0,:, kk]

        ax[kk].plot(avg[q,0,:, kk], ratio, label=f'{s:.2f}')
        ax[kk].set_xscale('log')

    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'Ratio of Distances')
    ax[kk].set_title(f'Position/Redshift Space {knn}-NN Distance Ratio')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('zcdf_cdf_ratio_results.png')
