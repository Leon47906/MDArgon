#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.units import temperature

kB = 1.38064852e-23
Mass = 39.948*1.66053906660e-27
Sigma = 0.33916e-9
Epsilon = 137.9

def plot(filename):
    # Load data
    f = open(filename, 'r')
    # read from first line the system size
    system_size = float(f.readline())
    # read from second line the initial temperature
    t_init = float(f.readline())
    # read from third line the number of particles
    N = int(f.readline())
    # read from fourth line the number of steps
    steps = int(f.readline())
    # read from fifth line the resolution
    resolution = int(f.readline())
    dt = float(f.readline())
    steps = steps//resolution
    e_kin = np.zeros(steps)
    virial = np.zeros(steps)
    for idx, line in enumerate(f):
        step = idx//(N+2)
        data = line.split()
        if len(data) == 2:
            e_kin[step] = float(data[1])/N
        elif len(data) == 1:
            virial[step] = float(data[0])*Epsilon*kB
        else:
            continue
    f.close()
    # rescale the energy
    e_kin = e_kin*Epsilon
    # initialize the temperature as a function of time
    #production_start = int(4*10e4/resolution)
    production_start = 0
    temperatures = np.zeros(steps-production_start)
    pressures = np.zeros(steps-production_start)
    for i in range(0, steps-production_start):
        temperatures[i] = 2/3*np.mean(e_kin[production_start:production_start+i+1])
        pressures[i] = kB*N*temperatures[i]/system_size**3 + np.mean(virial[production_start:production_start+i+1])/(3*system_size**3)
        pressures[i] *= 1e-9
    dt = 1*10e15 # dt in seconds
    dt *= np.sqrt(Mass*Sigma**2/Epsilon/kB)
    time = np.linspace(0, steps*dt, steps-production_start)
    time /= 1000 # convert to ps
    # Create plot of the temperature and pressure, with a red line at the mean temperature and pressure
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(14, 4)
    axs[0].plot(time, temperatures, label='Temperature in K')
    axs[0].set_xlabel('Time in ps')
    axs[0].set_ylabel('Temperature in K')
    axs[0].legend()
    axs[1].plot(time, pressures, label='Pressure in Gpa')
    axs[1].set_xlabel('Time in ps')
    axs[1].set_ylabel('Pressure in GPa')
    plt.savefig('macro_plot.png')
    plt.close()
    print(f"Plot saved as 'macro_plot.png")

if __name__ == '__main__':
    filename = 'data.txt'
    plot(filename)


