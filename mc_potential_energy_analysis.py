#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.cluster.hierarchy import average
from sympy.physics.mechanics import potential_energy
from sympy.physics.units import avogadro_number


'''
def energies(filename="MCdata.txt"):
    # Load data
    with open(filename, 'r') as f:
        # Read metadata
        system_size = float(f.readline())
        T_init = float(f.readline())
        N = int(f.readline())
        sweeps = int(f.readline())
        seed = int(f.readline())

        # Initialize e_pot array
        e_pot = list()

        # Process lines in the file
        step = -1  # Track step number
        for idx, line in enumerate(f):
            if idx % (N + 2) == 0:
                step += 1
                if step >= sweeps:
                    break

            # Skip header or empty lines
            data = line.split()
            if len(data) == 1 or len(data) == 3:
                continue

            # Store energy in e_pot
            e_pot.append(float(data[0]))
    return np.array(e_pot)
'''

def energies(filename="MCdata.txt"):
    # Load data
    with open(filename, 'r') as f:
        # Read data
        system_size = float(f.readline())
        T_init = float(f.readline())
        N = int(f.readline())
        sweeps = int(f.readline())
        resolution = int(f.readline())
        seed = int(f.readline())

        # Initialize e_pot array
        e_pot = np.zeros(sweeps)

        # Process lines in the file
        for step, line in enumerate(f):
            if step >= sweeps:
                break
            data = line.split()
            e_pot[step] = float(data[0])
    return e_pot


def plot_energy(data, filename="mc_energies.png"):
    '''
    nsweeps = np.zeros(len(data))
    mean_data = np.zeros(len(data))
    for idx, i in enumerate(data):
        nsweeps[idx] = idx
        mean_data[idx] = np.mean(data[:idx+1])
    # Plot the data
    plt.plot(nsweeps,mean_data, label='Potential energy in kB')
    '''
    plt.plot(data, label=r'Potential energy in $\epsilon$')
    plt.xlabel('Sweeps')
    plt.ylabel(r'Potential energy in $\epsilon$')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'")

def binning_analysis_with_variance(data):
    """
    Perform binning analysis with variance estimation for different bin sizes.

    Parameters:
        data (np.ndarray): Array of average potential energies.

    Returns:
        dict: Dictionary containing bin sizes and estimated variances.
    """
    n = len(data)
    max_bin_size = 2**int(np.floor(np.log2(n)))  # Maximum bin size as a power of 2
    results = {'bin_size': [], 'variance': []}
    total_variance = np.var(data)
    bin_size = 1
    while bin_size <= max_bin_size:
        # Group data into bins
        num_bins = n // bin_size
        truncated_data = data[:num_bins * bin_size]  # Truncate data to match the bin size
        binned_data = truncated_data.reshape((num_bins, bin_size)).mean(axis=1)

        # Calculate variance estimate
        mean_square = np.mean(binned_data**2)
        square_mean = np.mean(binned_data)**2
        variance = np.sqrt(np.abs(mean_square - square_mean))
        # Store results
        results['bin_size'].append(bin_size)
        results['variance'].append(bin_size*variance/total_variance)
        # Increase bin size
        bin_size *= 2
    return results

def save_binning_variance_plot(results, filename="binning_variance_plot.png"):
    """
    Save a plot of the variance as a function of bin size.

    Parameters:
        results (dict): Dictionary containing variancebin sizes and variances.
        filename (str): Filename for the saved plot.
    """
    bin_sizes = results['bin_size']
    variances = results['variance']

    plt.figure(figsize=(8, 6))
    plt.plot(bin_sizes, variances, marker='o', linestyle='-', color='blue')
    plt.xscale('log', base=2)
    plt.xlabel(r"Bin size $M$", fontsize=14)
    plt.ylabel(r"$\Delta O_M$", fontsize=14)
    plt.title("Binning Analysis: Variance vs Bin Size", fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as '{filename}'")

def binning_analysis(V_array,bin_size):
    N=len(V_array)
    d=N//bin_size
    V_M=np.zeros(d)
    v_M=np.mean(V_array)
    for j in range(0,d):
        V_M[j]=(np.mean(V_array[j*bin_size:(j+1)*bin_size])-v_M)**2
    return np.mean(V_M)*bin_size/np.var(V_array)

def binning_plot(V_results):
    splits = np.log2(len(V_results))
    x = np.array([2**i for i in range(np.ceil(splits).astype(int))])

    y = np.zeros(len(x))
    for i in range(0,len(x)):
        y[i]=binning_analysis(V_results,x[i])

    fig, ax = plt.subplots()
    ax.plot(x,y)
    ax.set_xscale("log", base=2)
    plt.grid()
    plt.savefig("binning_plot.png")
    print("Plot saved as 'binning_plot.png'")


def plot_means(data, resolution=1, correlation_time=1, filename="means.png"):
    plot_length = len(data)//resolution
    means = np.zeros(plot_length)
    errors = np.zeros(plot_length)
    for idx,i in enumerate(range(0, len(data), resolution)):
        means[idx] = np.mean(data[:i+1])
        errors[idx] = np.std(data)/np.sqrt((i+1)/correlation_time)
    # Plot the means with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(resolution*np.arange(plot_length), means, yerr=3*errors, fmt='+', color='blue', ecolor='red', capsize=5)
    plt.xlabel("Sweeps", fontsize=14)
    plt.ylabel(r"Mean Potential Energy per particle in $\epsilon$", fontsize=14)
    plt.title("Mean Potential Energy vs Sweeps", fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as '{filename}'")

# Example usage
if __name__ == "__main__":
    # Example data
    potential_energies = energies()
    print(np.var(potential_energies))
    #analysis_results = binning_analysis_with_variance(potential_energies)
    #save_binning_variance_plot(analysis_results)
    binning_plot(potential_energies)
    #plot_energy(potential_energies)
    plot_means(potential_energies,5000,120)
