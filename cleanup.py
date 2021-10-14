import os
import glob
#import yt
import numpy as np
#yt.enable_parallelism()

dirs = glob.glob('Halos/FoF/fiducial/*/')

#import pdb; pdb.set_trace()
#print(len(dirs))

zero = 0
one = 0
two = 0
three = 0
four = 0

for dir in dirs:

    zero  += os.stat(f'{dir}/groups_000/group_tab_000.0').st_size
    one   += os.stat(f'{dir}/groups_001/group_tab_001.0').st_size
    two   += os.stat(f'{dir}/groups_002/group_tab_002.0').st_size
    #three += os.stat(f'{dir}/groups_003/group_tab_003.0').st_size
    #four  += os.stat(f'{dir}/groups_004/group_tab_004.0').st_size

    try:
        os.system(f'rm -rf {dir}/groups_003')
        os.system(f'rm -rf {dir}/groups_004')
        print('deleted', dir)
    except:
        print('failed or already deleted', dir)

    print('done with', dir)

#sizes = np.array([zero, one, two, three, four])

#np.save('Quijote_size', sizes)

print(f'Zeroth snapshots have a total size of {zero:.3e} bytes')
print(f'First snapshots have a total size of {one:.3e} bytes')
print(f'Second snapshots have a total size of {two:.3e} bytes')
print(f'Third snapshots have a total size of {three:.3e} bytes')
print(f'Fourth snapshots have a total size of {four:.3e} bytes')

