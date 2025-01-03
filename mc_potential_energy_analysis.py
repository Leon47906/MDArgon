#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.cluster.hierarchy import average
from sympy.physics.mechanics import potential_energy


def energies(filename):
    kB = 1.38064852e-23
    # Load data
    f = open(filename, 'r')
    # read from first line the number of particles
    system_size = float(f.readline())
    N = int(f.readline())
    sweeps = int(f.readline())
    resolution = int(f.readline())
    sweeps = sweeps//resolution
    e_pot = np.zeros(sweeps)
    for idx, line in enumerate(f):
        data = line.split()
        e_pot[idx] = float(data[0])/kB
    f.close()
    return e_pot


def plot_energy(data, filename="energies.png"):
    plt.plot(data, label='Potential energy in kB')
    plt.xlabel('Sweeps')
    plt.ylabel('Potential energy in kB')
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
    print(total_variance)
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

def plot_means(data, resolution=1, correlation_time=1, filename="means.png"):
    plot_length = len(data)//resolution
    means = np.zeros(plot_length)
    errors = np.zeros(plot_length)
    for idx,i in enumerate(range(0, len(data), resolution)):
        means[idx] = np.mean(data[:i+1])
        errors[idx] = np.std(data)/np.sqrt((i+1)/correlation_time)
    # Plot the means with error bars
    plt.figure(figsize=(8, 6))
    plt.errorbar(resolution*np.arange(plot_length), means, yerr=3*errors, fmt='-', color='blue', ecolor='red', capsize=5)
    plt.xlabel("Sweeps", fontsize=14)
    plt.ylabel("Mean Potential Energy (kB)", fontsize=14)
    plt.title("Mean Potential Energy vs Sweeps", fontsize=16)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as '{filename}'")

# Example usage
if __name__ == "__main__":
    # Example data
    potential_energies = energies("MCdata.txt")
    analysis_results = binning_analysis_with_variance(potential_energies)
    save_binning_variance_plot(analysis_results)
    plot_means(potential_energies,200,1)
