import shelve

with shelve.open('DirectionalSuite/00000') as db:
    cCDF = db['cCDF']
    print(len(cCDF),len(cCDF[0]))

    LOS = cCDF[0]
    perp = cCDF[1]
    for q in range(len(LOS)):
        print(LOS[q].k, LOS[q].cdf(10), perp[q].cdf(10))

