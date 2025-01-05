#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

def plot(filename):
    kB = 1.38064852e-23
    # Load data
    name = filename.split('.')[0]
    f = open(filename, 'r')
    # read from first line the system sizes
    system_sizes = f.readline().split()
    # read from second line the temperatures
    temperatures = f.readline().split()
    n_s = len(system_sizes)
    n_t = len(temperatures)
    energies = np.zeros((n_s, n_t))
    std_devs = np.zeros((n_s, n_t))
    for idx, line in enumerate(f):
        data = line.split()
        s = int(data[0])
        t = int(data[1])
        energies[s, t] = float(data[2])/kB
        std_devs[s, t] = float(data[3])/kB
    f.close()
    # Create plot
    fig, ax = plt.subplots()
    fig.figsize = (12, 6)
    for i in range(n_s):
        energy_list = energies[i, :]
        std_dev_list = 2*std_devs[i, :]
        ax.errorbar(temperatures, energy_list, yerr=std_dev_list, label=f'{float(system_sizes[i])*1e9} nm')
    ax.set_xlabel('Temperature in K')
    ax.set_ylabel('Potential energy in kB')
    ax.legend()
    plt.savefig(f'{name}.png')
    plt.close()
    print(f"Plot saved as '{name}.png")

if __name__ == '__main__':
    plot('potentialEnergies.txt')
