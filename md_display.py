import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def make_movie(filename):
    kB = 1.38064852e-23
    # Load data
    f = open(filename, 'r')
    # read from first line the number of particles
    system_size = float(f.readline())
    N = int(f.readline())
    sweeps = int(f.readline())
    x = np.zeros((N,sweeps))
    y = np.zeros((N,sweeps))
    z = np.zeros((N,sweeps))
    for idx, line in enumerate(f):
        data = line.split()
        # load the x, y, z coordinates into the arrays
        x[idx%N, idx//N] = float(data[0])
        y[idx%N, idx//N] = float(data[1])
        z[idx%N, idx//N] = float(data[2])
    f.close()
    # Create animation, plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Set axis limits (adjust as needed)
    ax.set_xlim(0, system_size)
    ax.set_ylim(0, system_size)
    ax.set_zlim(0, system_size)

    # Initialize the scatter plot
    sc = ax.scatter(x[:, 0], y[:, 0], z[:, 0], c='b', marker='o')

    # Update function for animation
    def update(frame):
        sc._offsets3d = (x[:, frame], y[:, frame], z[:, frame])
        return sc,

    # Create the animation, which shows every 50th frame
    ani = FuncAnimation(fig, update, frames=range(0,sweeps,50), blit=True)
    # Save the animation as a video file
    output_file = 'Monte_Carlo_animation.mp4'
    writer = FFMpegWriter(fps=30, metadata={'artist': 'Matplotlib'}, bitrate=2400)
    ani.save(output_file, writer=writer)

    print(f"Animation saved as {output_file}")

filename = "MCdata.txt"
make_movie(filename)