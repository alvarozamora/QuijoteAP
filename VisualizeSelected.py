import numpy as np
import matplotlib.pyplot as plt
import io
import imageio as IO
from PIL import Image



def VisualizeSelection(pts, title="Visualize_Selected", verbose=False):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(*pts)
    bufs = []
    for i in range(0,360,1):

        if i % 5 == 0 and verbose:
            print(i)
    
        ax.view_init(10, i)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        bufs.append(buf)

    if verbose:
        print("Making MP4")

    with IO.get_writer(f'{title}.mp4', fps=36) as writer:

        for buf in bufs:
            #image = IO.imread(buf)
            #import pdb; pdb.set_trace()
            writer.append_data(np.array(Image.open(buf)))
        writer.close()

if __name__ == "__main__":

    selected_points = np.random.uniform(size=(3,10**3))

    VisualizeSelection(selected_points, title='Uniform_1e3', verbose=True)
