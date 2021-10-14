import readgadget
import readfof
import redshift_space_library as RSL
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils import *
import pdb
import copy

sim = 3
part_path = f'/oak/stanford/orgs/kipac/users/arkab/quijote_scratch/quijote_data/fiducial/sim_{sim}/snap_003'
head = readgadget.header(part_path)
hubble   = head.Hubble
print("hubble is", hubble)
redshift = head.redshift
axis = 2
sim_path = f'/oak/stanford/orgs/kipac/users/arkab/quijote_scratch/quijote_data/fiducial/sim_{sim}/'


snap_number = 3
boxsize = 1e3
FoF = readfof.FoF_catalog(sim_path, int(snap_number),
                                  long_ids=False,
                                  swap=False, SFR=False,
                                  read_IDs=False)
pos = FoF.GroupPos/1e3#/1e3            #Halo positions in Mpc/h, a second /1e3 normalizes to boxsize = 1
mass  = FoF.GroupMass*1e10        #Halo masses in solar masses
vel = FoF.GroupVel



print(f"pos min/max = {pos.min():.4f}/{pos.max():.4f}")

#rands = np.random.uniform(size=(10**5,3)).astype(np.float32)*boxsize
rands = np.random.uniform(size=(10**6,3))*boxsize
print(f"Rands min/max = {rands.min():.4f}/{rands.max():.4f}")

print("Real/Random Shape", pos.shape, rands.shape)
#XYZDProfile(pos, rands, 'Reals',bs=boxsize)


# Shift Datapoints
rpos = copy.deepcopy(pos)
RSL.pos_redshift_space(pos, vel,
                       boxsize, hubble,
                       redshift, axis)
print(f"zpos min/max = {pos.min():.4f}/{pos.max():.4f}")
print("z/Random Shape", pos.shape, rands.shape)
#XYZDProfile(pos, rands, 'PosShift',bs=boxsize)

# Shift Randoms for z-kNN
'''
rrands = copy.deepcopy(rands)
RSL.pos_redshift_space(rands, np.zeros_like(pos,dtype=np.float32),
                       boxsize, hubble,
                       redshift, axis)
print("z/zRandom Shape", pos.shape, rands.shape)
#XYZDProfile(pos, rands, 'AllShift')


ATN(rpos,pos,rrands,rands,'ATN')
'''
ATN(rpos.astype(np.float64),pos.astype(np.float64),rands,rands,'ATN',bs = 1e3)


CDFs(rpos.astype(np.float64), pos.astype(np.float64), rands, '1NN', bs=boxsize,c=10)
