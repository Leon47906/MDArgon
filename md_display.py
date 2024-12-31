import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.cluster.hierarchy import average


def make_movie(filename):
    kB = 1.38064852e-23
    # Load data
    f = open(filename, 'r')
    # read from first line the number of particles
    system_size = float(f.readline())
    N = int(f.readline())
    sweeps = int(f.readline())
    resolution = int(f.readline())
    sweeps = sweeps//resolution
    '''
    x = np.zeros((N,sweeps))
    y = np.zeros((N,sweeps))
    z = np.zeros((N,sweeps))
    e_pot = np.zeros(sweeps)
    for idx, line in enumerate(f):
        step = idx//(N+1)
        idx = idx % (N+1)
        data = line.split()
        if len(data) == 1:
            e_pot[step] = float(data[0])/kB
        else:
            # load the x, y, z coordinates into the arrays
            x[idx, step] = float(data[0])
            y[idx, step] = float(data[1])
            z[idx, step] = float(data[2])
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
    ani = FuncAnimation(fig, update, frames=sweeps, blit=True)
    # Save the animation as a video file
    output_file = 'Monte_Carlo_animation.mp4'
    writer = FFMpegWriter(fps=30, metadata={'artist': 'Matplotlib'}, bitrate=2400)
    ani.save(output_file, writer=writer)

    print(f"Animation saved as {output_file}")
    '''
    e_pot = np.zeros(sweeps)
    for idx, line in enumerate(f):
        data = line.split()
        e_pot[idx] = float(data[0])/kB
    fig2, axs = plt.subplots(1, 1)
    axs.plot(e_pot, label='Potential energy in kB')
    axs.set_xlabel('# of sweeps')
    axs.set_ylabel('Energy')
    fig2.savefig('Monte_Carlo_energy.svg')
    average_energy = np.mean(e_pot)
    print(f"Average energy: {average_energy} kB")

filename = "MCdata.txt"
make_movie(filename)