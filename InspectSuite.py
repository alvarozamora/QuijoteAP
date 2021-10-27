import shelve
from tqdm import tqdm


dir = 'EqualSkySuite_OmpStretch/'
it = tqdm(range(15000))
#it = tqdm(range(500))

for i in it:
    it.set_description(f"Processing {i}")
    try:
        #print(f"DirectionalSuite/{i:05d}")
        #print(f"{i}")
        with shelve.open(f"{dir}{i:05d}") as db:

            cCDF = db['cCDF']
            #print(len(cCDF),len(cCDF[0]))

            LOS = cCDF[0]
            perp = cCDF[1]

            '''
            for q in range(len(LOS)):
                print(LOS[q].k, LOS[q].cdf(10), perp[q].cdf(10))
            '''
    except:
        print(f"missing {i}")

