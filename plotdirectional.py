import shelve

from matplotlib.pyplot import tight_layout
import numpy as np
from utils import *

# Load Results
x = []; z = []
zx = []; zz = []

done = glob.glob('results_2_dir/*.npz'); 

xcdfs  = np.zeros((len(done), 6, 2, 199, 6))
zcdfs = np.zeros((len(done), 6, 2, 199, 6))
zxcdfs  = np.zeros((len(done), 6, 2, 199, 6))
zzcdfs = np.zeros((len(done), 6, 2, 199, 6))

failed = 0
for q, sim in enumerate(done):
    try: 
        with np.load(sim) as data:
            if data['pcdfs'].shape == (0,):
                print(f'{sim} is bad')
                continue
            x.append(data['pcdfs'])
            z.append(data['zcdfs'])
            zx.append(data['zpcdfs'])
            zz.append(data['zzcdfs'])

            print(f"Read {q+1} of {len(done)}")
    except:
        print(f"Failed reading {q+1} of {len(done)}")
        failed += 1

x = np.array(x)
z = np.array(z)
zx = np.array(zx)
zz = np.array(zz)

p_stdev =  x.std(0)/np.sqrt(len(x))
p_ztdev = zx.std(0)/np.sqrt(len(zx))
p_avg  =  x.mean(0)
p_zavg = zx.mean(0)

z_stdev =  z.std(0)/np.sqrt(len(z))
z_ztdev = zz.std(0)/np.sqrt(len(zz))
z_avg  =  z.mean(0)
z_zavg = zz.mean(0)

print(len(x), len(z), len(zx), len(zz), failed)

S = [1, 1.01, 1.02, 1.03, 1.05, 0.98]
k = [1, 2, 3, 4, 8]
k = np.arange(32)+1
k = 2**np.arange(8)-1
k = 2**np.arange(13)
k = 2**np.arange(6)


print("Plotting unstretched CDFs")
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S[:1]):

        y_pcdf = p_avg[q,1,:, kk]
        y_zcdf = z_avg[q,1,:,kk]
        x_pcdf = p_avg[q,0,:, kk]
        x_zcdf = z_avg[q,0,:,kk]

        ax[kk].loglog(x_pcdf, y_pcdf, '.-', label=f'{s:.2f} (x)')
        ax[kk].loglog(x_zcdf, y_zcdf, '.--', label=f'{s:.2f} (z)')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF')
    ax[kk].set_title(f'{knn}NN Peaked CDF')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
    ax[kk].set_ylim(1e-3,1)
plt.tight_layout()
plt.savefig('cdf_results_dir.png')

print("Plotting perpendicular CDF ratios perp real/redshift")
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        y_pcdf = p_avg[q,1,:, kk]
        y_zcdf = p_zavg[q,1,:,kk]
        x_pcdf = p_avg[q,0,:, kk]
        x_zcdf = p_zavg[q,0,:,kk]

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf, '.-', label=f'{s:.2f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/zp)')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('cdf_results_dir_xx_ratio.png')

print("Plotting perpendicular CDF ratios parallel real/redshift")
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        y_pcdf = z_avg[q,1,:, kk]
        y_zcdf = z_zavg[q,1,:,kk]
        x_pcdf = z_avg[q,0,:, kk]
        x_zcdf = z_zavg[q,0,:,kk]

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf, '.-', label=f'{s:.2f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (z/zz)')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('cdf_results_dir_zz_ratio.png')

print("Plotting CDF ratios in real space, perp/parallel")
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        y_pcdf = p_avg[q,1,:, kk]
        y_zcdf = z_avg[q,1,:,kk]
        x_pcdf = p_avg[q,0,:, kk]
        x_zcdf = z_avg[q,0,:,kk]

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf, '.-', label=f'{s:.2f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/z)')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('cdf_results_dir_xz_ratio_realspace.png')

print("Plotting CDF ratios in redshiftspace")
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        y_pcdf = p_zavg[q,1,:, kk]
        y_zcdf = z_zavg[q,1,:,kk]
        x_pcdf = p_zavg[q,0,:, kk]
        x_zcdf = z_zavg[q,0,:,kk]

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf, '.-', label=f'{s:.2f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/z)')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('cdf_results_dir_xz_ratio_redshiftspace.png')

print("Plotting CDF ratios in redshiftspace renormalized")
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        y_pcdf = p_avg[q,1,:, kk]
        y_zcdf = z_avg[q,1,:,kk]
        x_pcdf = p_avg[q,0,:, kk]
        x_zcdf = z_avg[q,0,:,kk]

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf, f'C{q}--', label=f'real {s:.2f}', alpha=0.5)
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/z)')
    ax[kk].grid(alpha=0.3)

for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        y_pcdf = p_zavg[q,1,:,kk]
        y_zcdf = z_zavg[q,1,:,kk]
        x_pcdf = p_zavg[q,0,:,kk]
        x_zcdf = z_zavg[q,0,:,kk]

        denom_x = p_zavg[0,0,:, kk]
        denom_z = z_zavg[0,0,:,kk]
        denom = denom_x/denom_z

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf/denom, f'C{q}.-', label=f'renorm {s:.2f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/z)')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('cdf_results_dir_xz_ratio_redshiftspace_renorm.png')

print("Plotting CDF ratios in redshiftspace renormalized (zoom)")
zoom = 85
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        y_pcdf = p_avg[q,1,zoom:,kk]
        y_zcdf = z_avg[q,1,zoom:,kk]
        x_pcdf = p_avg[q,0,zoom:,kk]
        x_zcdf = z_avg[q,0,zoom:,kk]

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf, f'C{q}--', label=f'real {s:.2f}', alpha=0.5)
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/z)')
    ax[kk].grid(alpha=0.3)

for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        y_pcdf = p_zavg[q,1,zoom:,kk]
        y_zcdf = z_zavg[q,1,zoom:,kk]
        x_pcdf = p_zavg[q,0,zoom:,kk]
        x_zcdf = z_zavg[q,0,zoom:,kk]

        denom_x = p_zavg[0,0,zoom:, kk]
        denom_z = z_zavg[0,0,zoom:,kk]
        denom = denom_x/denom_z

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf/denom, f'C{q}.-', label=f'renorm {s:.2f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/z)')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('cdf_results_dir_xz_ratio_redshiftspace_renormzoom.png')

print("Plotting CDF ratios in redshiftspace renormalized (fake)")
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        y_pcdf = p_avg[q,1,:, kk]
        y_zcdf = z_avg[q,1,:,kk]
        x_pcdf = p_avg[q,0,:, kk]
        x_zcdf = z_avg[q,0,:,kk]

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf, f'C{q}--', label=f'real {s:.2f}', alpha=0.5)
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/z)')
    ax[kk].grid(alpha=0.3)

for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        
        fakeS = s/S[1]

        y_pcdf = p_zavg[q,1,:,kk]
        y_zcdf = z_zavg[q,1,:,kk]
        x_pcdf = p_zavg[q,0,:,kk]
        x_zcdf = z_zavg[q,0,:,kk]

        denom_x = p_zavg[1,0,:,kk]
        denom_z = z_zavg[1,0,:,kk]
        denom = denom_x/denom_z

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf/denom, f'C{q}.-', label=f'renorm {fakeS:.3f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/z)')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('cdf_results_dir_xz_ratio_redshiftspace_fakerenorm.png')

print("Plotting CDF ratios in redshiftspace renormalized (fake, zoom)")
zoom = 85
fig, ax = plt.subplots(len(k), 1, figsize=(20,10*len(k)))
for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        y_pcdf = p_avg[q,1,zoom:,kk]
        y_zcdf = z_avg[q,1,zoom:,kk]
        x_pcdf = p_avg[q,0,zoom:,kk]
        x_zcdf = z_avg[q,0,zoom:,kk]

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf, f'C{q}--', label=f'real {s:.2f}', alpha=0.5)
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/z)')
    ax[kk].grid(alpha=0.3)

for kk, knn in enumerate(k):
    for q, s in enumerate(S):
        
        fakeS = s/S[1]

        y_pcdf = p_zavg[q,1,zoom:,kk]
        y_zcdf = z_zavg[q,1,zoom:,kk]
        x_pcdf = p_zavg[q,0,zoom:,kk]
        x_zcdf = z_zavg[q,0,zoom:,kk]

        denom_x = p_zavg[1,0,zoom:,kk]
        denom_z = z_zavg[1,0,zoom:,kk]
        denom = denom_x/denom_z

        ax[kk].semilogx(x_pcdf, x_pcdf/x_zcdf/denom, f'C{q}.-', label=f'renorm {fakeS:.3f}')
    ax[kk].set_xlabel(r'Distance (Mpc $h^{-1}$)')
    ax[kk].set_ylabel(f'{knn}NN Peaked CDF Ratio')
    ax[kk].set_title(f'{knn}NN Peaked CDF Ratio (p/z)')
    ax[kk].grid(alpha=0.3)
    ax[kk].legend()
plt.tight_layout()
plt.savefig('cdf_results_dir_xz_ratio_redshiftspace_fakerenormzoom.png')

assert False, "End Here"

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