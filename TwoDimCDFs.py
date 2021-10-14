from FastCDF import fastCDF
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np



def TwoDimensionalCDF(perp, los, high, Ng = 50, verbose=False):

    # Weights
    y = np.ones_like(los)

    # Define Grids for CDF
    xg = np.linspace(0, high, Ng)
    yg = np.linspace(0, high, Ng)

    # Total Data and Excluded Data 
    total = len(perp) + len(los); excluded = (perp > high).sum() + (los > high).sum()
    if verbose:
        print(f"Total Data = {total}; Excluded = {excluded}.")
    if excluded/total > 0.1:
        print("WARNING: MORE THAN 10% OF DATA IS EXCLUDED FROM 2D-CDF")

    # Find CDF Using fastCDF 
    cdf = fastCDF([perp, los], [xg, yg], y).reshape(Ng,Ng)

    return cdf, xg, yg

def VisualizeTheCDF(cdf, xg, yg, target='CDF_2D', tight=False):

    # Take Gradients -- Second Derivative is PDF
    grad = np.gradient(cdf,xg, axis=0)
    der2  = np.gradient(grad.reshape(len(xg),len(yg)),yg, axis=1)
    ga = der2

    # Compute Peaked CDF
    pcdf = np.minimum(cdf,1-cdf) # peaked CDF

    plt.rcParams["mpl_toolkits.legacy_colorbar"] = False
    
    # Grid 0: CDF
    fig = plt.figure(figsize=(15,8))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.45,cbar_mode="each",cbar_location="right",cbar_pad=0.1)
    cc = grid[0].imshow(cdf,origin='lower', extent=[xg.min(),xg.max(),yg.min(),yg.max()])
    grid[0].contour(cdf,origin='lower', extent=[xg.min(),xg.max(),yg.min(),yg.max()],colors="white",alpha=.5,
                        levels=np.linspace(0,1,11))
    grid[0].set_title("CDF")
    grid.cbar_axes[0].colorbar(cc)
    grid[0].set_xlabel(r'Perpendicular Distance (Mpc/$h$)')
    grid[0].set_ylabel(r'Parallel Distance (Mpc/$h$)')


    c = grid[1].imshow(pcdf,origin='lower', extent=[xg.min(),xg.max(),yg.min(),yg.max()],vmax=1)
    grid[1].contour(pcdf,origin='lower', extent=[xg.min(),xg.max(),yg.min(),yg.max()],colors="white",alpha=.5,
                        levels=np.linspace(0,1,11))
    grid[1].set_title("peaked CDF")
    grid.cbar_axes[1].colorbar(c)
    grid[1].set_xlabel(r'Perpendicular Distance (Mpc/$h$)')

    ccc = grid[2].imshow(ga,origin='lower', extent=[xg.min(),xg.max(),yg.min(),yg.max()])
    # NEW LINE
    contour = False
    if contour:
        grid[2].contour(ga,origin='lower', extent=[xg.min(),xg.max(),yg.min(),yg.max()],colors="white",alpha=.5,
                        levels=np.linspace(ga.min(),ga.max(),11))
                        #levels=np.logspace(np.log10(ga[ga>0].min()),np.log10(ga.max()),11))

    grid[2].set_title(r"PDF = $\partial^2 CDF/\partial x \partial y$")
    grid.cbar_axes[2].colorbar(ccc)
    grid[2].set_xlabel(r'Perpendicular Distance (Mpc/$h$)')


    if tight:
        plt.tight_layout()

    if type(target) == str:
        plt.savefig(f'{target}.png')
    else:
        # Intended for io.BytesIO()-like object
        plt.savefig(target)

if __name__ == '__main__':
    
    from scipy.spatial import cKDTree

    data = np.random.uniform(size=(10**5, 3))
    rand = np.random.uniform(size=(10**6, 3))
    tree = cKDTree(data, leafsize=1, balanced_tree=True, compact_nodes=True,boxsize=1)

    r, ids = tree.query(rand, k=1, workers=-1)

    abs_r = np.abs(data[ids]-rand)
    abs_r = np.minimum(abs_r,1-abs_r)

    #perp = np.sort(np.sqrt(abs_r[:,0]**2 + abs_r[:,1]**2))
    #perp = np.sqrt(np.sort((abs_r[:,0])**2 + np.sort(abs_r[:,1])**2))
    perp = abs_r[:,0]
    z = abs_r[:,2]

    # Unit Tests
    print(f"Inputs: perp.shape = {perp.shape}; z.shape = {z.shape}")
    VisualizeTheCDF(*TwoDimensionalCDF(perp, z, 0.1/4, verbose=True), 'TestFastCDF')




