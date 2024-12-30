import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def plot_data(filename):
    kB = 1.38064852e-23
    # Load data
    f = open(filename, 'r')
    # read from first line the number of particles
    system_size = float(f.readline())
    N = int(f.readline())
    steps = int(f.readline())
    resolution = int(f.readline())
    steps = steps//resolution
    x = np.zeros((N,steps))
    y = np.zeros((N,steps))
    z = np.zeros((N,steps))
    e_pot = np.zeros(steps)
    e_kin = np.zeros(steps)
    for idx, line in enumerate(f):
        step = idx//(N+1)
        idx = idx % (N+1)
        data = line.split()
        if len(data) == 2:
            e_pot[step] = float(data[0])/kB
            e_kin[step] = float(data[1])/kB
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
    ani = FuncAnimation(fig, update, frames=steps, blit=True)
    # Save the animation as a video file
    output_file = 'particles_animation.mp4'
    writer = FFMpegWriter(fps=30, metadata={'artist': 'Matplotlib'}, bitrate=2400)
    ani.save(output_file, writer=writer)

    print(f"Animation saved as {output_file}")
    fig2, axs = plt.subplots(1, 2)
    axs[0].plot(e_pot, label='Potential energy in kB')
    axs[0].set_xlabel('Time step')
    axs[0].set_ylabel('Energy')
    axs[0].legend()
    axs[1].plot(e_kin, label='Kinetic energy in kB')
    axs[1].set_xlabel('Time step')
    axs[1].set_ylabel('Energy')
    axs[1].legend()
    fig2.figsize = (15, 5)
    # Save the plot as an svg file
    output_file = 'energy_plot.svg'
    fig2.savefig(output_file)
    average_potential_energy = np.mean(e_pot)
    print(f"Time-average potential energy: {average_potential_energy*kB} J")
    pressure = N*kB*140/(system_size)**3*1e-5
    print(f"Pressure: {pressure} bar")
    return None

# Example
filename = 'data.txt'
plot_data(filename)