#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def plot_data(filename):
    kB = 1.38064852e-23
    # Load data
    f = open(filename, 'r')
    # read from first line the number of particles
    system_size = float(f.readline())
    temperature = float(f.readline())
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
    ani = FuncAnimation(fig, update, frames=steps, blit=False)
    # Save the animation as a video file
    output_file = 'particles_animation.mp4'
    writer = FFMpegWriter(fps=30, metadata={'artist': 'Matplotlib'}, bitrate=2400)
    ani.save(output_file, writer=writer)

    print(f"Animation saved as {output_file}")
    return None

def plot_energies(filename):
    kB = 1.38064852e-23
    Mass = 39.948*1.66053906660e-27
    Sigma = 0.33916e-9
    Epsilon = 137.9*kB
    # Load data
    f = open(filename, 'r')
    # read from first line the number of particles
    system_size = float(f.readline())
    temperature = float(f.readline())
    N = int(f.readline())
    steps = int(f.readline())
    resolution = int(f.readline())
    steps = steps//resolution
    e_int = np.zeros(steps)
    e_kin = np.zeros(steps)
    for idx, line in enumerate(f):
        step = idx//(N+1)
        idx = idx % (N+1)
        data = line.split()
        if len(data) == 2:
            e_int[step] = float(data[0])/N
            e_kin[step] = float(data[1])/N
        else:
            continue
    f.close()
    # Create the time axis with correct scaling to fs
    # std::sqrt(Mass*Dalton*Sigma*Sigma*nm*nm/(Epsilon*kB))
    dt = 0.1*np.sqrt(Mass*Sigma**2/Epsilon)/10e-15
    time = np.linspace(0, steps*dt, steps)
    # Create plot of energies seperately
    fig, axs = plt.subplots(1,3)
    fig.set_size_inches(15, 5)
    axs[0].plot(time, e_int, label='Interaction energy')
    axs[0].set_xlabel('Time in fs')
    axs[0].set_ylabel('Interaction energy per particle in $\epsilon$')
    axs[0].legend()
    axs[1].plot(time, e_kin, label='Kinetic energy')
    axs[1].set_xlabel('Time in fs')
    axs[1].set_ylabel('Kinetic energy per particle in $\epsilon$')
    axs[1].legend()
    axs[2].plot(time, e_int+e_kin, label='Total energy')
    axs[2].set_xlabel('Time in fs')
    axs[2].set_ylabel('Total energy per particle in $\epsilon$')
    plt.savefig('energies.png')
    plt.close()
    print(f"Plot saved as 'energies.png")


if __name__ == '__main__':
    #filename = 'MCdata.txt'
    filename = 'data.txt'
    #plot_data(filename)
    plot_energies(filename)