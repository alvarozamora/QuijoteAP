import numpy as np
import matplotlib
from numpy.lib.arraysetops import _isin_dispatcher
matplotlib.use('agg')
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree as Tree
from time import time
import pdb
import os
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import copy
import pandas
from colossus.cosmology import cosmology
import glob
#from VisualizeSelected import VisualizeSelection
#from TwoDimCDFs import VisualizeTheCDF, TwoDimensionalCDF
#from fast_interp import interp1d
from scipy.interpolate import interp1d
from SEdist import SE_distribution

import shelve

def stretch(data, zdata, query, boxsize, s):

	# When stretching one axis by s, other two are compressed by sqrt(s) to preserve volume
	other = np.sqrt(s)
	
	# Stretch Data
	sdata = copy.deepcopy(np.array([data[:,0]/other, data[:,1]/other, data[:,2]*s]).T)

	# Stretch zData
	szdata = copy.deepcopy(np.array([zdata[:,0]/other, zdata[:,1]/other, zdata[:,2]*s]).T)
	
	# Stretch Query
	squery = copy.deepcopy(np.array([query[:,0]/other, query[:,1]/other, query[:,2]*s]).T)

	# Stretch Box
	sbs = np.array([boxsize/other, boxsize/other, boxsize*s])

	return sdata, szdata, squery, sbs

def extract_results(directory='results_2/',results={},c2=False):
	# Average over sims
	cdfs = []; zcdfs = []
	cumulants = []; zcumulants = []
	zpresent = False
	done = glob.glob(directory+'/*.npz'); zpresent=True;

	#cdfs  = np.zeros((len(done), 6, 2, 199, 6))
	#zcdfs = np.zeros((len(done), 6, 2, 199, 6))
	if c2:
		c2s = np.zeros((len(done), 6, 2, 200))
		zc2s = np.zeros((len(done), 6, 2, 200))
	for q, sim in enumerate(done):
		with np.load(sim) as data:
			if data['cdfs'].shape == (0,):
				print(f'{sim} is bad')
				continue
			#if data['cdfs'].shape == (6, 2, 200):
			else:
				cdfs.append(data['cdfs'])
				zcdfs.append(data['zcdfs'])
				print(len(cdfs))
			#cdfs[q] = data['cdfs']
			#zcdfs[q] = data['zcdfs']
			if c2:
				c2s[q] = data['c2']
				zc2s[q] = data['zc2']
			print(f"Read {q+1} of {len(done)}")
	'''
	for key in results.keys():
		
		try:
			results[key][0][0].shape
		except:
			continue
		
		assert(results[key][0][0].shape == (6, 2, 99, 128))

		#import pdb; pdb.set_trace()
		cdfs.append(results[key][0][0])
		cumulants.append(results[key][0][1])

		try:
			zcdfs.append(results[key][0][2])
			zcumulants.append(results[key][0][3])
			zpresent = True
		except:
			pass
	'''
	'''
	if zpresent:
		return np.array(cdfs), np.array(cumulants), np.array(zcdfs), np.array(zcumulants)
	else:
		return np.array(cdfs), np.array(cumulants)
	'''
	if c2:
		cdfs, zcdfs, c2s, zc2s
	else:
		return np.array(cdfs), np.array(zcdfs)
		


def sigma(r):

        # Nearest Neighbors
        one = r[:,0]
        two = r[:,1]
        onecdf = twocdf = np.arange(1, len(two)+1)/(len(two))

        # Inner 95 of one and two
        xone_inner = xtwo_inner = np.logspace(np.log10(5), np.log10(20), 100)
        yone_inner = interp1d(np.sort(one), onecdf,bounds_error=False,fill_value=0)(xone_inner)
        #import pdb; pdb.set_trace()
        ytwo_inner = interp1d(np.sort(two), twocdf,bounds_error=False,fill_value=0)(xtwo_inner)#, bounds_error=None)
        
        # Moments
        #import pdb; pdb.set_trace()
        # Arka Tom 2020
        nV = - 2 * ( np.log(1 - yone_inner) + 1/2 * (yone_inner - ytwo_inner)/(1 - yone_inner))
        s2 = - 2 * ( np.log(1 - yone_inner) + 1/1 * (yone_inner - ytwo_inner)/(1 - yone_inner))/nV**2

        assert (s2>=0).all(), f'Negative Variance'
        return xone_inner, s2

def second_cumulant(knn):

	# Define spatial grid 
	low = 5; high = 50; # Mpc/h
	r = np.logspace(np.log10(low), np.log10(high), 200)

	# Define CDF interpolant based on len(knn) = number of randoms
	emp_cdf = np.arange(1, len(knn)+1)/(len(knn))
	
	cdf = np.concatenate((np.ones_like(r).reshape(len(r),1), np.array([np.interp(r, np.sort(knn[:,k]), emp_cdf,left=0,right=1) for k in range(knn.shape[-1])]).T), axis=-1)
	
	# Define CIC distributions 
	cic = -np.diff(cdf, axis=-1)

	# Define k array for k, k**2 term
	k = np.arange(knn.shape[-1])[None,:]

	# Compute Second Cumulant
	C2 = (k**2*cic).sum(-1)/cic.sum(-1) - ((k*cic).sum(-1)/cic.sum(-1))**2

	return np.array([r, C2])

def second_cumulant_analytic(cosmo, rgrid, n,z=0.5):

	# Define Cosmology
	try:
		cosmo = cosmology.setCosmology(cosmo)
	except:
		print("Failed to initialize specified cosmology. Falling back to planck18.")
		cosmo = cosmology.setCosmology("planck18")

	# Define r --> Volume map
	V = lambda r0: 4*np.pi*r0**3/3

	# Define window function
	W = lambda r,r0: 8*np.pi**2 * ((2*r0 - r)**3/12 + r*(2*r0-r)**2/12)

	# Integrand for second cumulant
	integrand = lambda r0: lambda r: r*r*cosmo.correlationFunction(r,z)*W(r,r0)

	# Do all 
	# The 1e-3 lower limit and 500 upper limit comes from a limitation in the colossus package in evaluating the correl func
	integrals = np.array([integrate.quad(integrand(r0), 1e-3, 2*r0)[0] for r0 in rgrid])

	# Scale and move to get result
	result = n*V(rgrid) + n*n*integrals

	return result

def ReadData(simulation, Nrand=10**6, nmass = 10**5, snap=2, vel_factor=1.0):
	import readgadget
	import readfof
	import redshift_space_library as RSL
	import copy

	#part_path = simulation+f'/snap_{snap:03d}';
	header_path = f'headers/snap_{snap:03d}'
	sim_path = simulation
	head = readgadget.header(header_path)
	hubble   = head.Hubble
	redshift = head.redshift

	axis = 2

	boxsize = 1e3
	FoF = readfof.FoF_catalog(sim_path, int(snap),
                                  long_ids=False,
                                  swap=False, SFR=False,
                                  read_IDs=False)
	pos = FoF.GroupPos/1e3            # Halo positions in Mpc/h, a second /1e3 normalizes to boxsize = 1
	mass  = FoF.GroupMass*1e10        # Halo masses in solar masses
	vel = FoF.GroupVel*vel_factor     # km/s

	cont = True
	if nmass !=0 :
		#assert len(mass) > nmass, f"not enough for {nmass} subsample"
		if len(mass) < nmass:
			print(f" len(mass) > nmass, not enough for {nmass} subsample")
			cont = False
		idxs = mass.argsort()[-nmass:][::-1]
		pos = pos[idxs]
		vel = vel[idxs]
		
	fix = True
	if pos.max()/boxsize >= 1:
		
		print("loaded a thing that has a thing on the edge or outside the box")
		if not fix:
			cont = False
		else:
			pos = pos % boxsize
			assert pos.max() < boxsize, "Didn't fix"


	
	# Axis for RSD
	rpos = copy.deepcopy(pos)
	RSL.pos_redshift_space(pos, vel,
                       boxsize, hubble,
                       redshift, axis)

	rands = np.random.uniform(size=(Nrand,3))*boxsize

	
	if pos.max()/boxsize >= 1 and cont:
		print("loaded a thing that has a thing on the edge or outside the box after RSDs")
		if not fix:
			cont = False
		else:
			pos = pos % boxsize

	return rpos, pos, vel, rands, mass, hubble, redshift, boxsize, cont

def APdistance(r,vel,H,axis,z):
	return np.array([H*r*np.eye(3)[axis]])

def kNNs(data, query, K=np.unique(np.floor(np.logspace(np.log10(1.01),np.log10(2**12+0.1),50)).astype(int)),bs=1,verb=True, getids=False):

	assert (data.shape[-1]==3) & (query.shape[1]==3), 'wrong format'

	if verb:
		print("Generating Tree")
	start = time()
	try:
		tree = Tree(data, leafsize=min(K.max(),256), compact_nodes=True, balanced_tree=True, boxsize=bs)
		if verb:
			print(f"Generated Tree in {time()-start:.3f} seconds")
		if verb:
			print("Querying")
		start = time()
		r, ids = tree.query(query, k=K, workers=-1)
		if verb:
			print(f"Queried in {time()-start:.3f} seconds")

		if getids:
			return r, ids
		else:
			return r
	except:
		print(data.max(), data.min(), bs)
	

	

def CDF(r):
	cdf = np.arange(1,r.shape[0]+1)/r.shape[0]
	cdf[cdf>0.5] = 1 - cdf[cdf>0.5]
	return cdf

def XYZD(data, query, r, ids,avg=True,c=0,bs=1,verb=True, low=None, high=None):

	if type(bs) != float:
		bs = np.array(bs).reshape(1,1,3)

	if verb:
		print("Computing XYZD")
		start = time()


	D = np.abs(data[ids]-query[:,None])
	D = np.minimum(D, bs-D)
	if low is not None and high is not None:
		assert r.shape[-1] == 1
		filter = (r >= low) & (r <= high)
		D = D[filter].reshape(filter.sum(),1,3)
	
	#x = np.sort(D[:,:,0],axis=0)
	#y = np.sort(D[:,:,1],axis=0)
	#z = np.sort(D[:,:,2],axis=0)
	x = D[:,:,0]
	y = D[:,:,1]
	z = D[:,:,2]

	ratio = np.sort(D[:,:,0]/D[:,:,2],axis=0)
	D = np.linalg.norm(D,axis=-1)

	if avg==True:
		x = x.mean(axis=0)
		y = y.mean(axis=0)
		z = z.mean(axis=0)
		D = D.mean(axis=0)

	#pdb.set_trace()
	if verb:
		print(f"Computed XYZD in {time()-start:.3f} seconds")


	return x, y, z, D, ratio

def SquarePyramid(data, query, r, ids, k=8, bs=1, verb=True):

	if type(bs) != float:
		bs = np.array(bs).reshape(1,1,3)

	if verb:
		print("Computing SquarePyramid XYZD")
		start = time()

	D = np.abs(data[ids]-query[:,None])
	D = np.minimum(D, bs-D)

	which = np.argmax(np.abs(D),axis=2)

	x = np.sort(np.array([np.sort(np.abs(D[i][w==0][:,0]))[:k] for i, w in enumerate(which)]),0)
	y = np.sort(np.array([np.sort(np.abs(D[i][w==1][:,1]))[:k] for i, w in enumerate(which)]),0)
	z = np.sort(np.array([np.sort(np.abs(D[i][w==2][:,2]))[:k] for i, w in enumerate(which)]),0)

	assert (x.shape == (D.shape[0], k)) & (y.shape == (D.shape[0], k)) & (z.shape == (D.shape[0], k)), 'Not enough neighbors in Square Pyramid'

	#import pdb; pdb.set_trace()
	p = (x + y)/2

	#p = np.sort(np.concatenate((x,y),0),0)

	return p, z

def equalsky(data, query, r, ids, k=8, bs=1, verb=True):

	if type(bs) != float:
		bs = np.array(bs).reshape(1,1,3)

	if verb:
		print("Computing EqualSky XYZD")
		start = time()

	D = np.abs(data[ids]-query[:,None])
	D = np.minimum(D, bs-D)
	
	#which = np.argmax(np.abs(D),axis=2)
	cr = np.sqrt(D[:,:,0]**2 + D[:,:,1]**2)
	which = cr/np.abs(D[:,:,2]) < np.tan(np.pi/3)

	
	box1 = [np.sort(r[i][w==True ],axis=-1)[:k] for i, w in enumerate(which)]
	box2 = [np.sort(r[i][w==False],axis=-1)[:k] for i, w in enumerate(which)]

	Box1 = []
	Box2 = []

	off = 0
	for box in box1:
		if len(box) != k:
			off += 1
		else:
			Box1.append(box)

	off2 = 0
	for box in box2:
		if len(box) != k:
			off2 += 1
		else:
			Box2.append(box)

	cutoff = np.min([len(Box1), len(Box2)])
	Box1 = np.array(Box1)[:cutoff]
	Box2 = np.array(Box2)[:cutoff]

	print(f"There are {off} off entries.")
	print(f"There are {off2} off2 entries.")
	assert cutoff > 0.9*D.shape[0], "Not enough neighbors in band"
	assert Box1.shape[1] == Box2.shape[1] and Box1.shape[1] == k, "not right number of Nearest Neighbors"


	#box1 = np.sort(np.array([np.sort(r[i][w==True],axis=-1)[:k] for i, w in enumerate(which)]),0)  # Line of Sight
	#box2 = np.sort(np.array([np.sort(r[i][w==False],axis=-1)[:k] for i, w in enumerate(which)]),0) # Perpendicular

	#assert (box1.shape == (D.shape[0], k)) & (box2.shape == (D.shape[0], k)), 'Not enough neighbors in band'
	#if np.random.uniform() > 0.99:
	#	print(box1.shape, box2.shape)

	#import pdb; pdb.set_trace()
	#p = (x + y)/2

	#p = np.sort(np.concatenate((x,y),0),0)
	#return box2, box1
	#print(Box2[:,0].mean(), Box1[:,0].mean())
	return Box2, Box1

def PZD(data, query, r, ids,avg=True,c=0,bs=1,verb=True, low=None, high=None):

	x, y, z, D, ratio = XYZD(data, query, r, ids, avg=avg,c=c,bs=bs,verb=verb, low=low, high=high)

	#p = (x + y)/2
	p = x

	return p, z, D

def XYZDProfile(data, query, Name, K=np.unique(np.floor(np.logspace(np.log10(1.01),np.log10(2**12+0.1),50)).astype(int)),bs=1):

	x, y, z, D, ratio = XYZD(data, query, *kNNs(data, query, K, bs),bs=bs)
	p = (x + y)/2
	p = x
	print("Plotting XYZD Profile")
	plt.figure(figsize=(6,5))

	#plt.loglog(K,x,'.--',label=r'$x$')
	#plt.loglog(K,y,'.--',label=r'$y$')
	plt.loglog(K,p,'.--',label=r'$p$')
	plt.loglog(K,z,'.--',label=r'$z$')
	#plt.loglog(K,D,'.--',label=r'$D$')
	#plt.loglog(K,D/np.sqrt(3),'.--',label=r'$D/\sqrt{3}$')

	#plt.loglog(K,p/z,'.--', label=r'$p/z$')

	plt.xlabel("K")
	#plt.ylabel("p/z")
	plt.ylabel("Distance")
	plt.grid(alpha=0.3)
	plt.legend()

	plt.tight_layout()
	plt.savefig(f"{Name}.png",dpi=230)

# All together now
def ATN(data, zdata, query, zquery, Name, K=np.unique(np.floor(np.logspace(np.log10(1.01),np.log10(2**12+0.1),50)).astype(int)),bs=1):

	#Real Data Real Randoms
	print("Real Data Real Randoms")
	x, y, z, D, ratio = XYZD(data, query, *kNNs(data, query, K, bs),bs=bs)
	p = (x + y)/2
	p = x
	#z Data Real Randoms
	print("z Data Real Randoms")
	zx, zy, zz, zD, zratio = XYZD(zdata, query, *kNNs(zdata, query, K, bs),bs=bs)
	zp = (zx + zy)/2
	zp = zx
	#z data z randoms
	#print("z Data z Randoms")
	#zzx, zzy, zzz, zzD = XYZD(zdata, zquery, *kNNs(zdata, zquery, K, bs))
	#zzp = (zzx + zzy)/2
	#z data z randoms
	print("z Data z Data")
	zzx, zzy, zzz, zzD, zzratio = XYZD(zdata, zdata, *kNNs(zdata, zdata, K+1, bs),bs=bs)
	zzp = (zzx + zzy)/2
	zzp = zzx

	print("Plotting All Together")
	plt.figure(figsize=(6,5))

	plt.loglog(K,ratio.mean(axis=0),'.--', label=r'$RD$', alpha=0.5)
	plt.loglog(K,zratio.mean(axis=0),'.--', label=r'$RD_z$', alpha=0.5)
	plt.loglog(K,zzratio.mean(axis=0),'.--', label=r'$D_zD_z$', alpha=0.5)

	plt.xlabel("K")
	plt.ylabel("p/z")
	plt.grid(alpha=0.3)
	plt.legend()

	plt.tight_layout()
	plt.savefig(f"{Name}_L.png",dpi=230)

	plt.figure(figsize=(6,5))

	plt.loglog(z,p/z,'.--', label=r'$RD$', alpha=0.5)
	plt.loglog(z,zp/zz,'.--', label=r'$RD_z$', alpha=0.5)
	plt.loglog(z,zzp/zzz,'.--', label=r'$D_zD_z$', alpha=0.5)

	plt.xlabel("z (Mpc "+r"$h^{-1}$)")
	plt.ylabel("p/z")
	plt.grid(alpha=0.3)
	plt.legend()

	plt.tight_layout()
	plt.savefig(f"{Name}_E.png",dpi=230)


def CDFs(data, zdata, query, Name, K=np.array([1,4]), bs=1,c=0):

	RDkNN = kNNs(data, query, K, bs)
	RDzkNN = kNNs(zdata, query, K, bs)
	DzDzkNN = kNNs(zdata, zdata, K+1, bs)


	RDx, _, RDz, _, _ = XYZD(data, query, *RDkNN, avg=False,bs=bs)
	RDzx, _, RDzz, _, _ = XYZD(zdata, query, *RDzkNN, avg=False,bs=bs)
	DzDzx, _, DzDzz, _, _ = XYZD(zdata, zdata, *DzDzkNN, avg=False,bs=bs)

	RDx = [RDx[:,i][RDkNN[0][:,i]>c] for i in range(2)]
	RDz = [RDz[:,i][RDkNN[0][:,i]>c] for i in range(2)]
	RDzx = [RDzx[:,i][RDzkNN[0][:,i]>c] for i in range(2)]
	RDzz = [RDzz[:,i][RDzkNN[0][:,i]>c] for i in range(2)]
	DzDzx = [DzDzx[:,i][DzDzkNN[0][:,i]>c] for i in range(2)]
	DzDzz = [DzDzz[:,i][DzDzkNN[0][:,i]>c] for i in range(2)]

	#1NN
	plt.figure(figsize=(9,8))
	plt.subplot(211)
	plt.loglog(RDx[0], CDF(RDx[0]),'C0--', label=r'$RD$-x')
	plt.loglog(RDz[0], CDF(RDz[0]),'C0-', label=r'$RD$-z')

	plt.loglog(RDzx[0], CDF(RDzx[0]), 'C1--', label=r'$RD_z$-x')
	plt.loglog(RDzz[0], CDF(RDzz[0]), 'C1-', label=r'$RD_z$-z')

	plt.loglog(DzDzx[0], CDF(DzDzx[0]), 'C2--', label=r'$D_zD_z$-x')
	plt.loglog(DzDzz[0], CDF(DzDzz[0]), 'C2-',  label=r'$D_zD_z$-z')
	plt.xlabel("Distance (Mpc "+r"$h^{-1}$)")
	plt.ylabel('CDF')
	plt.xlim(1e-1,1e2)
	plt.ylim(1e-2,1)
	plt.grid(alpha=0.3)
	plt.legend()

	#4NN
	plt.subplot(212)
	plt.loglog(RDx[1], CDF(RDx[1]),'C0--', label=r'$RD$-x')
	plt.loglog(RDz[1], CDF(RDz[1]),'C0-', label=r'$RD$-z')

	plt.loglog(RDzx[1], CDF(RDzx[1]), 'C1--', label=r'$RD_z$-x')
	plt.loglog(RDzz[1], CDF(RDzz[1]), 'C1-', label=r'$RD_z$-z')

	plt.loglog(DzDzx[1], CDF(DzDzx[1]), 'C2--', label=r'$D_zD_z$-x')
	plt.loglog(DzDzz[1], CDF(DzDzz[1]), 'C2-',  label=r'$D_zD_z$-z')
	plt.xlabel("Distance (Mpc "+r"$h^{-1}$)")
	plt.ylabel('CDF')
	plt.xlim(1e-1,1e2)
	plt.ylim(1e-2,1)
	plt.grid(alpha=0.3)
	plt.legend()


	plt.savefig('1NN.png')

def APMeasurement(data, query, redshift=1, axis=2, bs=1, K = [1,4,64,256], Om = 0.3175,verb=True):

	kNN = kNNs(data, query, K, bs,verb=verb)

	x, y, z, D, ratio = XYZD(data, query, *kNN, avg=False,c=0,bs=bs,verb=verb)

	#std and mean of kNN Distances
	sD = D.std(axis=0)
	D = D.mean(axis=0)

	#std and mean of perpendicular/LOS distances
	p = (x+y)/2
	sp = p.std(axis=0)
	p = p.mean(axis=0)#*(1+redshift)
	los = z.mean(axis=0)

	for q, k in enumerate(K,0):
		dz_zdt = Y(redshift,Om)*los[q]/p[q]
		print(f"{k}-AP Measurement(({D[q]:.4f} +/- {sD[q]:.4f}) Mpc/h ) = {dz_zdt:.4f}")



def Y(z, Om):
	Ol = 1-Om
	def integrand(x):
		return np.sqrt( (Om*(1+z)**3 + Ol) / (Om*(1+x)**3 + Ol) )

	result = integrate.quad(integrand, 0, z)

	assert(result[1]<1e-6)
	return result[0]/z

def X(z, Om):
	Ol = 1-Om

	return 1/np.sqrt(Om*(1+z)**3 + Ol)


def CDF_percentile(r, p=np.arange(5,95+1)):
	r = np.percentile(r,p,axis=0)
	p = p/100
	p[p> 0.5] = 1 - p[p>0.5]
	assert (p.min()>=0) & (p.max() <= 0.5)
	return np.array([r, np.array([p for knn in range(r.shape[-1])]).T ])

def L2optimize(x,z):

	import scipy.optimize as optimize

	def f(a):
		return ((x-a*z)**2).mean()

	Opt = optimize.least_squares(f,1,bounds=(0.2,1.8))
	#pdb.set_trace()
	return Opt.x[0]


def MeanCutOff(data, query, redshift=1, axis=2, bs=1, K = [1,4,64,256], Om = 0.3175,verb=True,label='RD',nmass=0):

	r, ids = kNNs(data, query, K, bs,verb=verb)

	rmean = r.mean(axis=0)

	D = np.abs(data[ids]-query[:,None])
	D = np.minimum(D, bs-D)

	x = [D[:,q,0][r[:,q]>rmean[q]] for q in range(len(K))]
	#y = D[:,:,1]
	z = [D[:,q,2][r[:,q]>rmean[q]] for q in range(len(K))]

	xCDF = [CDF_percentile(X) for X in x]
	zCDF = [CDF_percentile(Z) for Z in z]

	for q in range(len(K)):
		print(f"Optimizing {K[q]}NN")

		print(L2optimize(xCDF[q][0],zCDF[q][0]))


	mini = np.min([xq[0].min() for xq in xCDF])*0.7
	maxi = np.max([xq[0].max() for xq in xCDF])*1.05


	for q, k in enumerate(K):
		plt.subplot(f"{len(K)}1{q+1}")
		plt.xlabel(r"Distance ($h^{-1}$ Mpc)")
		plt.ylabel(f"Peaked {k}NN-CDF")
		plt.loglog(*xCDF[q],label=f'{label}-x')
		plt.loglog(*zCDF[q], '--',label=f'{label}-z')
		plt.xlim(mini,maxi)
		plt.ylim(5e-2,1)
		plt.grid(True,alpha=0.3)
		plt.legend()


def MeanCutOff_Scale(data, query, redshift=1, axis=2, bs=1, K = [1,4,64,256], Om = 0.3175,verb=True,label='RD',nmass=0):

	r, ids = kNNs(data, query, K, bs,verb=verb)

	rmean = r.mean(axis=0)

	D = np.abs(data[ids]-query[:,None])
	D = np.minimum(D, bs-D)

	x = [D[:,q,0][r[:,q]>rmean[q]] for q in range(len(K))]
	#y = D[:,:,1]
	z = [D[:,q,2][r[:,q]>rmean[q]] for q in range(len(K))]

	xCDF = [CDF_percentile(X) for X in x]
	zCDF = [CDF_percentile(Z) for Z in z]

	scales = []
	for q in range(len(K)):
		print(f"Optimizing {K[q]}NN")

		scales.append(L2optimize(xCDF[q][0],zCDF[q][0]))
	print(scales)
	zCDF = [(zCDF[q][0]*scales[q],zCDF[q][1]) for q in range(len(K))]

	mini = np.min([xq[0].min() for xq in xCDF])*0.7
	maxi = np.max([xq[0].max() for xq in xCDF])*1.05


	for q, k in enumerate(K):
		plt.subplot(f"{len(K)}1{q+1}")
		plt.xlabel(r"Distance ($h^{-1}$ Mpc)")
		plt.ylabel(f"Peaked {k}NN-CDF")
		plt.loglog(*xCDF[q],label=f'{label}-x')
		plt.loglog(*zCDF[q], '--',label=f'{label}-z')
		plt.xlim(mini,maxi)
		plt.ylim(5e-2,1)
		plt.grid(True,alpha=0.3)
		plt.legend()

def Count(data, query, redshift=1, axis=2, bs=1, K = [1,4,64,256], Om = 0.3175,verb=True,label='RD',nmass=0):

	r, ids = kNNs(data, query, K, bs,verb=verb)

	rmean = r.mean(axis=0)

	D = np.abs(data[ids]-query[:,None])
	D = np.minimum(D, bs-D)

	x = [D[:,q,0][r[:,q]>rmean[q]] for q in range(len(K))]

	before = len(np.unique(ids[:,0]))
	after  = len(np.unique(ids[r[:,0]>rmean[0],0]))

	aftertotal = len(x[0])
	total = D.shape[0]
	print(f"Total = {total}; Total After = {aftertotal}; Unique Before Cut = {before}; Unique After Cut = {after}")

def L2(data, query, redshift=1, axis=2, bs=1, K = [1,4,64,256], Om = 0.3175,verb=True,label='',style=''):

	r, ids = kNNs(data, query, K, bs,verb=verb)

	rmean = r.mean(axis=0)

	D = np.abs(data[ids]-query[:,None])
	D = np.minimum(D, bs-D)

	x = [D[:,q,0][r[:,q]>rmean[q]] for q in range(len(K))]
	#y = D[:,:,1]
	z = [D[:,q,2][r[:,q]>rmean[q]] for q in range(len(K))]

	xCDF = [CDF_percentile(X) for X in x]
	zCDF = [CDF_percentile(Z) for Z in z]

	mini = np.min([xq[0].min() for xq in xCDF])*0.7
	maxi = np.max([xq[0].max() for xq in xCDF])*1.05

	l2 = np.array([((xCDF[q][0]-zCDF[q][0])**2) for q in range(len(K))])
	L2mean = [ll2.mean() for ll2 in l2]

	for q, k in enumerate(K):
		plt.subplot(f"{len(K)}1{q+1}")
		plt.xlabel(r"Distance ($h^{-1}$ Mpc)")
		plt.ylabel(f"L2 of {k}PCDF")
		plt.loglog(xCDF[q][0],l2[q],style, label=f'L2-'+label)
		plt.xlim(mini,maxi)
		plt.grid(True,alpha=0.3)
		plt.legend()

	print(L2mean)


def DistanceFilter(low, high, perp, LOS, ids=None):
	
	filter = (perp > low) & (perp < high) & (LOS > low) & (LOS < high)

	assert filter.sum() < len(perp)


	if ids is None:
		return perp[filter].reshape(filter.sum(),1), LOS[filter].reshape(filter.sum(),1)
	else:
		return perp[filter].reshape(filter.sum(),1), LOS[filter].reshape(filter.sum(),1), ids[filter]


def CanLoad(file):
	
	try:
		np.load(file)
		return True
	except:
		return False

def minimize_loss(d, z, threeD=True):

	'''
	This function finds the optimal scaling parameter such that < z > / < d > is close to 1 (or 1/2).
	If d is the 3D distance, then it should be close to 1/2, so include the factor of 2.
	If d is a component of 3d displacement vector, then it should be 1, so excluse the factor of 2.

	Parameters
	----------------------------------------------------------------------------------------------------
	z: array-like
	  z-component of vector
	d: array-like
	  x-component of vector or L2 distance of vector.

	Returns
	----------------------------------------------------------------------------------------------------
	a: instance of scipy minimize class
	  a contains all of the relevant information, including information regarding number of iterations
	  and convergence status. a.x[0] returns a float value.
	'''

	if threeD:
		loss = lambda a : np.abs(2*a*z.sum()/d.sum() - 1)
	else:
		loss = lambda a : np.abs(  a*z.sum()/d.sum() - 1)

	from scipy.optimize import minimize

	a = minimize(loss,0.98)

	return a

def APFactor(Om, z):

	# Initialize True Cosmology
	quijote = {'flat': True, 'H0': 67.11, 'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624, }
	cosmology.addCosmology('Quijote', quijote)
	quijote = cosmology.setCosmology('Quijote')

	# Initialize Probe Cosmology
	probeparams = {'flat': True, 'H0': 67.11, 'Om0': Om, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624, }
	cosmology.addCosmology('Probe', probeparams)
	probe = cosmology.setCosmology('Probe')

	# Compute AP Factor	
	AP = quijote.angularDiameterDistance(z)*quijote.Hz(z) / (probe.angularDiameterDistance(z)*probe.Hz(z))

	return AP

def sFactor(Om, z):

	AP = APFactor(Om, z)

	return AP**(2/3)


def getA(data, query, boxsize, lo=None, hi=None, verbose=True):

		# Generate Tree
		dtree = Tree(data, leafsize=1, compact_nodes=True, balanced_tree=True, boxsize=boxsize)

		# Query at Tree
		r, ids = dtree.query(query, k=[1], n_jobs=-1)

		if verbose:
				print(f"Unique IDs = {len(np.unique(ids))} of {len(data)} data points.")
				print(f"Mean is {r.mean(0)[0]:.2f} Mpc/h")

		# Displacement Vector (abs)
		D = np.abs(data[ids]-query[:,None])
		D = np.minimum(D, np.array(boxsize).T[None,:]-D)


		# LOS Distances
		assert (D.shape == (len(query), 1, 3)), "Shape not as expected"
		LOS = D[:,:,2]

		# Perpendicular Distances
		perp = D[:,:,0]

		# Filter, if true
		if lo is not None and hi is not None:
				filter = (r[:,0] >= lo) & (r[:,0] <= hi)

				if np.random.randn() > 2:
					ycdf = np.arange(1,len(r)+1)/(len(r)+1)
					xcdf = np.sort(r[:,0])
					lo_hi_cdf = np.interp([lo, hi], xcdf, ycdf)

					print(f"CDF(lo) = {lo_hi_cdf[0]:.3f}; CDF(hi) = {lo_hi_cdf[1]:.3f}")

				if verbose:
						print(f"Filter netted {filter.sum()} out of {len(perp)} ({100*filter.sum()/len(perp):.2f}%)")
				perp = perp[filter]  
				LOS  = LOS[filter]
		else:
				print("Not Filtering")
				perp = perp.T
				LOS = LOS.T

		a = minimize_loss(perp, LOS, False)
		assert a.success, "Did not converge"
		return a.x[0]


def getKS(data, query, boxsize, lo=None, hi=None, verbose=True):

		# Generate Tree
		#start = time()
		dtree = Tree(data, leafsize=1, compact_nodes=True, balanced_tree=True, boxsize=boxsize)
		#print(f"Done with tree in {time()-start:.3f} seconds.")

		# Query at Tree
		#start = time()
		r, ids = dtree.query(query, k=[1], n_jobs=-1)
		#print(f"Done querying tree in {time()-start:.3f} seconds.")

		if verbose:
				print(f"Unique IDs = {len(np.unique(ids))} of {len(data)} data points.")
				print(f"Mean is {r.mean(0)[0]:.2f} Mpc/h")

		# Displacement Vector (abs)
		#start = time()
		D = np.abs(data[ids]-query[:,None])
		D = np.minimum(D, np.array(boxsize).T[None,:]-D)
		#print(f"Done computing D in {time()-start:.3f} seconds.")


		# LOS Distances
		assert (D.shape == (len(query), 1, 3)), "Shape not as expected"
		start = time()
		LOS = np.sort(D[:,0,2])
		#print(f"Done computing and sorting LOS in {time()-start:.3f} seconds.")


		# Perpendicular Distances
		#start = time()
		perp = np.sort(D[:,0,0])
		#print(f"Done computing and sorting perp in {time()-start:.3f} seconds.")

		#Distances
		r = np.linspace(lo, hi, 10)
		cdf = np.arange(1, len(LOS)+1)/len(LOS)

		# fast_interp implementation (NOT DONE)
		#LOS = interp1d(0,100,)

		# scipy interpolation
		#mport pdb; pdb.set_trace()
		#LOS  = interp1d( LOS, cdf, bounds_error=False, fill_value=(0,1))
		#perp = interp1d(perp, cdf, bounds_error=False, fill_value=(0,1))
		#los =  LOS(r)
		#prp = perp(r)

		# numpy implementation
		#start = time()
		los = np.interp(r,  LOS, cdf)
		prp = np.interp(r, perp, cdf)
		#print(f"Done interpolating for KS in {time()-start:.3f} seconds.")
		
		
		#start = time()
		#s = len(LOS)//10
		#e = 2*len(LOS)//10+1
		#loss = np.abs(LOS[s:e:10000]-perp[s:e:10000]).mean()
		loss = np.abs(los-prp).mean()
		#print(loss)
		#print(f"Done computing loss in {time()-start:.3f} seconds.")



		return loss


def compressedCDF(data, query, boxsize, k, verbose=True):

		# Generate Tree
		#start = time()
		dtree = Tree(data, leafsize=np.max(k), compact_nodes=True, balanced_tree=True, boxsize=boxsize)
		#print(f"Done with tree in {time()-start:.3f} seconds.")

		# Query at Tree
		start = time()
		r, ids = dtree.query(query, k=k, n_jobs=-1)
		print(f"Done querying tree in {time()-start:.3f} seconds.")

		if verbose:
				print(f"Unique IDs = {len(np.unique(ids))} of {len(data)} data points.")
				print(f"Mean is {r.mean(0)[0]:.2f} Mpc/h")

		# Displacement Vector (abs)
		#start = time()
		D = np.abs(data[ids]-query[:,None])
		D = np.minimum(D, np.array(boxsize).T[None,:]-D)
		#print(f"Done computing D in {time()-start:.3f} seconds.")


		# LOS Distances
		assert (D.shape == (len(query), len(k), 3)), "Shape not as expected"
		#start = time()
		LOS = np.sort(D[:,:,2],axis=0)
		#print(f"Done computing and sorting LOS in {time()-start:.3f} seconds.")


		# Perpendicular Distances
		#start = time()
		perp = np.sort(D[:,:,0],axis=0)
		#print(f"Done computing and sorting perp in {time()-start:.3f} seconds.")

		#Distances
		cdf = np.arange(1, len(LOS)+1)/len(LOS)

		cLOS = []
		cprp = []

		for q, kk in enumerate(k):

			los = SE_distribution( LOS[:,q], compress="log", Ninterpolants=1000)
			los.k = kk
			cLOS.append(los)

			prp = SE_distribution(perp[:,q], compress="log", Ninterpolants=1000)
			prp.k = kk
			cprp.append(prp)

		return cLOS, cprp



def compressedequalCDF(data, query, boxsize, k, verbose=True, subsamples=1):

		# Split into subsamples
		# Ensure Random
		choice = np.random.permutation(np.arange(len(data)))    
		data = data[choice]
		datas = np.split(data, subsamples)
		queries = np.split(query, subsamples)
		
		r, ids = [], []
		for sub, data_ in enumerate(datas):
				
			try:
				# Generate Tree
				dtree = Tree(data_, leafsize=8*np.max(k), compact_nodes=True, balanced_tree=True, boxsize=boxsize)
			except:
				data_ = np.array(data_)
				boxsize = np.array(boxsize)

				which = (data_ != (data_%boxsize))
				print(which.sum())
				print(which.argmax())
				print(data_.flatten()[which.argmax()], boxsize)
				assert False, f"data min/max = {data_[data_.min(-1).argmin()]}, {data_[data_.max(-1).argmax()]}; boxsize = {boxsize}"
			#print(f"Done with tree in {time()-start:.3f} seconds.")

			# Query Tree
			r_, ids_ = dtree.query(queries[sub], k=8*np.max(k), n_jobs=-1)
			r.append(r_); ids.append(ids_ + sub*len(data)//subsamples)
		r = np.concatenate(r); ids = np.concatenate(ids)
		assert len(r) == len(query), f"{len(r)}, {len(query)}"
		if np.random.uniform() > 0.95:
			print(r[:,0].mean())

		perp, LOS = equalsky(data, query, r, ids, k=np.max(k), bs=boxsize, verb=True)

		cLOS = []
		cprp = []

		for q, kk in enumerate(k):

			los = SE_distribution( LOS[:,q], compress="log", Ninterpolants=1000)
			los.k = kk
			cLOS.append(los)

			prp = SE_distribution(perp[:,q], compress="log", Ninterpolants=1000)
			prp.k = kk
			cprp.append(prp)

		return cLOS, cprp


def minimize_loss(d, z, threeD=True):

        '''
        This function finds the optimal scaling parameter such that < z > / < d > is close to 1 (or 1/2).
        If d is the 3D distance, then it should be close to 1/2, so include the factor of 2.
        If d is a component of 3d displacement vector, then it should be 1, so excluse the factor of 2.

        Parameters
        ----------------------------------------------------------------------------------------------------
        z: array-like
                z-component of vector
        d: array-like
                x-component of vector or L2 distance of vector.

        Returns
        ----------------------------------------------------------------------------------------------------
        a: instance of scipy minimize class
                a contains all of the relevant information, including information regarding number of iterations
                and convergence status. a.x[0] returns a float value.
        '''

        if threeD:
                loss = lambda a : np.abs(2*a*z.sum()/d.sum() - 1)
        else:
                loss = lambda a : np.abs(  a*z.sum()/d.sum() - 1)

        from scipy.optimize import minimize

        A = minimize(loss, 0.98, method='Nelder-Mead')

        return A


if __name__ == "__main__":

	# Test AP Factor & sFactor
	Oms = [0.25, 0.3175, 0.4]
	for Om in Oms:
		print(f"APFactor(Om = {Om:.2f}, z = 1) = {APFactor(Om,1):.4f}; s = {sFactor(Om, 1):.4f}")

