#include "verlet.hpp"
#include <fstream>
#include <omp.h>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

std::array<double, 100> polarangles, azimuthalangles;
std::array<Vec3, 10000> vs;

void initialize_vs() {
    for (int i = 0; i < 100; i++) {
        polarangles[i] = M_PI * i / 100;
        azimuthalangles[i] = 2 * M_PI * i / 100;
    }
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 100; j++) {
            vs[i * 100 + j] = Vec3(std::sin(polarangles[i]) * std::cos(azimuthalangles[j]),
                                   std::sin(polarangles[i]) * std::sin(azimuthalangles[j]),
                                   std::cos(polarangles[i]));
        }
    }
}

std::vector<Vec3> randomDist(int N, double system_size,UniformRandomDouble *rd_ptr) {
    if (N <= 0) {
        throw std::invalid_argument("Number of atoms (N) must be greater than 0.");
    }
    if (system_size <= 0) {
        throw std::invalid_argument("System size must be greater than 0.");
    }
    UniformRandomDouble &rd = *rd_ptr;
    std::vector<Vec3> positions(N, Vec3());
    for (int i = 0; i < N; i++) {
        const double x = system_size * rd();
        const double y = system_size * rd();
        const double z = system_size * rd();
        positions[i] = Vec3(x, y, z);
    }
    return positions;
}

std::vector<Vec3> cubicLattice(int N, double system_size) {
    std::vector<Vec3> positions;

    if (N <= 0) {
        throw std::invalid_argument("Number of atoms (N) must be greater than 0.");
    }
    if (system_size <= 0) {
        throw std::invalid_argument("System size must be greater than 0.");
    }

    // Find the next cube root greater than or equal to N
    int cube_root = static_cast<int>(std::ceil(std::cbrt(N)));
    int cube_number = cube_root * cube_root * cube_root;

    // Calculate the lattice spacing
    double lattice_spacing = system_size / (cube_root);
    if (lattice_spacing <= 0) {
        throw std::runtime_error("Lattice spacing must be greater than 0.");
    }

    // Calculate the starting position for the lattice
    double center = system_size / 2.0;
    double start_position = center - (lattice_spacing * (cube_root - 1) / 2);

    // Generate positions on a cubic lattice
    int count = 0;
    for (int i = 0; i < cube_root && count < N; ++i) {
        for (int j = 0; j < cube_root && count < N; ++j) {
            for (int k = 0; k < cube_root && count < N; ++k) {
                double x = start_position + i * lattice_spacing;
                double y = start_position + j * lattice_spacing;
                double z = start_position + k * lattice_spacing;
                positions.emplace_back(x, y, z);
                count++;
            }
        }
    }
    return positions;
}

/*
std::tuple<double, std::vector<double>> acceptanceRate(const System &atom_system, const double &sum_of_potentials, const int &atom_idx, const Vec3 &new_position, const double &T) {
    const int N = atom_system.getN();
    const double system_size = atom_system.getSystemSize();
    const std::vector<double> &potentials = atom_system.getPotentialEnergies();
    std::vector<double> new_potentials = potentials, dpotentials(N, 0);
    int cell_index = atom_system.getCell(new_position);
    double sum_of_new_potentials = sum_of_potentials;
    for (int i : atom_system.getAtomsInCell(cell_index)) {
    	 if (i != atom_idx) {
            Vec3 position = atom_system.getAtom(i).getPosition();
            Vec3 diff_new = atom_system.PeriodicDifferece(new_position, position, system_size);
            new_potentials[i] = LennardJones(diff_new.norm2());
            dpotentials[i] = new_potentials[i] - potentials[i];
            sum_of_new_potentials += dpotentials[i];
        }
    }
    for (int i : atom_system.getAtomsInNeighboringCells(cell_index)) {
        Vec3 position = atom_system.getAtom(i).getPosition();
        Vec3 diff_new = atom_system.PeriodicDifferece(new_position, position, system_size);
        new_potentials[i] = LennardJones(diff_new.norm2());
        dpotentials[i] = new_potentials[i] - potentials[i];
        sum_of_new_potentials += dpotentials[i];
    }
    const double acceptance = std::exp(-(sum_of_new_potentials - sum_of_potentials) / (kB * T));
    std::pair<double, std::vector<double>> result(acceptance, dpotentials);
    return result;
}

void MC_step(System *atom_system_ptr, double *sum_of_potentials_ptr, const int &atom_idx, UniformRandomDouble *rd_ptr, const double &dr, const double &T
             , int *Naccept_ptr) {
    System &atom_system = *atom_system_ptr;
    UniformRandomDouble &rd = *rd_ptr;
    double &sum_of_potentials = *sum_of_potentials_ptr;
    const std::vector<double> &potential = atom_system.getPotentialEnergies();
    const double system_size = atom_system.getSystemSize();
    const Vec3 position = atom_system.getAtom(atom_idx).getPosition();
    //const int random_index = std::floor(10000 * rd());
    //const Vec3 velocity = dr * vs[random_index];
    //const Vec3 prop_position = atom_system.PeriodicPositionUpdate(position, velocity, 1.0);
    const Vec3 displacement = dr/std::sqrt(3) * Vec3(1-2*rd(), 1-2*rd(), 1-2*rd());
    const Vec3 prop_position = atom_system.PeriodicPositionUpdate(position, displacement, 1.0);
    double acceptance_rate;
    std::vector<double> dpotentials;
    tie(acceptance_rate, dpotentials) = acceptanceRate(atom_system, sum_of_potentials, atom_idx, prop_position, T);
    int &Naccept = *Naccept_ptr;
    if (rd() < acceptance_rate) {
        atom_system.updatePosition(atom_idx, prop_position);
        std::vector<double> new_potentials = potential;
        for (int i = 0; i < atom_system.getN(); i++) {
            new_potentials[i] += dpotentials[i];
            sum_of_potentials += dpotentials[i];
        }
        atom_system.updatePotentialEnergies(new_potentials);
        Naccept++;
    }
}

 */

__device__ void atomicAdd_double(double* address, double value)
{
    float old_lower = *reinterpret_cast<float*>(address);
    float old_upper = *(reinterpret_cast<float*>(address) + 1);
    float new_lower = old_lower + value;
    if (new_lower < old_lower) {
        old_upper += 1.0f;  // Handle carry-over
    }
    *reinterpret_cast<float*>(address) = new_lower;
    *(reinterpret_cast<float*>(address) + 1) = old_upper;
}

__global__ void MC_step_kernel(
    double *positions_x, double *positions_y, double *positions_z,
    double *potentials, double *sum_of_potentials,
    const int N, const double system_size, const double dr, const double T, int *Naccept, curandState *state) {

    // Get thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    curand_init(1234, idx, 0, &state[idx]);
    double rand_value = curand_uniform(&state[idx]);
    // Generate random displacement
    double dx = dr / sqrt(3.0) * (1.0 - 2.0 * (rand_value));
    double dy = dr / sqrt(3.0) * (1.0 - 2.0 * (rand_value));
    double dz = dr / sqrt(3.0) * (1.0 - 2.0 * (rand_value));

    // Proposed new position
    double new_x = positions_x[idx] + dx;
    double new_y = positions_y[idx] + dy;
    double new_z = positions_z[idx] + dz;

    // Periodic boundary conditions
    new_x = fmod(new_x + system_size, system_size);
    new_y = fmod(new_y + system_size, system_size);
    new_z = fmod(new_z + system_size, system_size);

    // Compute potential change
    double dE = 0.0;
    for (int j = 0; j < N; j++) {
        if (j != idx) {
            double dx = new_x - positions_x[j];
            double dy = new_y - positions_y[j];
            double dz = new_z - positions_z[j];

            // Apply periodic boundary conditions
            dx -= system_size * round(dx / system_size);
            dy -= system_size * round(dy / system_size);
            dz -= system_size * round(dz / system_size);

            double r2 = dx * dx + dy * dy + dz * dz;
            double new_potential = 4 * (pow(r2, -6) - pow(r2, -3));
            double old_potential = potentials[j];
            dE += new_potential - old_potential;
        }
    }

    // Metropolis criterion
    double acceptance = exp(-dE / (T));
    if (rand_value < acceptance) {
        positions_x[idx] = new_x;
        positions_y[idx] = new_y;
        positions_z[idx] = new_z;

        atomicAdd(Naccept, 1);
        atomicAdd_double(sum_of_potentials, dE);
    }
}

// CUDA Host Function
void MC_sweep_host(
    std::vector<double> &positions_x, std::vector<double> &positions_y, std::vector<double> &positions_z,
    std::vector<double> &potentials, double &sum_of_potentials, int N, double system_size,
    double dr, double T, int &Naccept) {

    // Allocate device memory
    double *d_positions_x, *d_positions_y, *d_positions_z;
    double *d_potentials, *d_sum_of_potentials;
    int *d_Naccept;
    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState) * N);  // Allocate space for N threads
    cudaMalloc(&d_positions_x, N * sizeof(double));
    cudaMalloc(&d_positions_y, N * sizeof(double));
    cudaMalloc(&d_positions_z, N * sizeof(double));
    cudaMalloc(&d_potentials, N * sizeof(double));
    cudaMalloc(&d_sum_of_potentials, sizeof(double));
    cudaMalloc(&d_Naccept, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_positions_x, positions_x.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions_y, positions_y.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions_z, positions_z.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_potentials, potentials.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sum_of_potentials, &sum_of_potentials, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Naccept, &Naccept, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 512;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int num_Sweeps = 10000;
    for (int sweep = 0; sweep < num_Sweeps; ++sweep) {
        MC_step_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_positions_x, d_positions_y, d_positions_z,
                                                       d_potentials, d_sum_of_potentials,
                                                       N, system_size, dr, T, d_Naccept, d_state);
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    cudaMemcpy(positions_x.data(), d_positions_x, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(positions_y.data(), d_positions_y, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(positions_z.data(), d_positions_z, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_of_potentials, d_sum_of_potentials, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Naccept, d_Naccept, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_positions_x);
    cudaFree(d_positions_y);
    cudaFree(d_positions_z);
    cudaFree(d_potentials);
    cudaFree(d_sum_of_potentials);
    cudaFree(d_Naccept);
    cudaFree(d_state);
}

void MC_sweep(System *atom_system_ptr, double *sum_of_potentials_ptr, UniformRandomDouble *rd_ptr, const double &dr
              , const double &T, int *Naccept_ptr) {
    System &atom_system = *atom_system_ptr;
    const int N = atom_system.getN();
    const double system_size = atom_system.getSystemSize();
    std::vector<double> positions_x(N), positions_y(N), positions_z(N);
    std::vector<double> potentials = atom_system.getPotentialEnergies();
    for (int i = 0; i < N; i++) {
        Vec3 position = atom_system.getAtom(i).getPosition();
        positions_x[i] = position.getX();
        positions_y[i] = position.getY();
        positions_z[i] = position.getZ();
    }
    double sum_of_potentials = *sum_of_potentials_ptr;
    int Naccept = *Naccept_ptr;
    MC_sweep_host(positions_x, positions_y, positions_z, potentials, sum_of_potentials, N, system_size, dr, T, Naccept);
    for (int i = 0; i < N; i++) {
        atom_system.getAtom(i).setPosition(Vec3(positions_x[i], positions_y[i], positions_z[i]));
    }
    atom_system.updatePotentialEnergies(potentials);
    *sum_of_potentials_ptr = sum_of_potentials;
    *Naccept_ptr = Naccept;
}

int main(int argc, char *argv[]) {
    if (argc != 1) {
        std::cout << "Usage: " << argv[0] << std::endl;
        return -1;
    }
    // parameters
    UniformRandomDouble random;
    //std::array<double, 10> system_sizes = {1.18419e-8, 9.39889e-9, 8.21068e-9, 7.4599e-9,
    //                                       6.92516e-9, 6.51682e-9, 6.19042e-9, 5.92093e-9,
    //                                       5.69297e-9, 5.4965e-9};
    //std::array<double, 8> system_sizes = {1.66e-8, 1.36e-8,1.18419e-8, 9.39889e-9, 8.21068e-9, 7.4599e-9, 6.92516e-9, 6.51682e-9};
    std::array<double, 2> system_sizes = {1.66e-8, 1.36e-8};
    int N = 1000;
    //std::array<double, 10> temperatures = {60, 90, 120, 150, 180, 210, 240, 270, 300, 330};
    //std::array<double, 8> temperatures = {20, 30, 50, 60, 90, 120, 150, 180};
    std::array<double, 2> temperatures = {20, 30};
    int sweeps = 10000;
    double dr = 0.165*nm;
    std::array<double, 4> potentialEnergies, stdDeviations;
    potentialEnergies.fill(0);
    stdDeviations.fill(0);
    //initialize_vs();
    for (int idx = 0; idx < potentialEnergies.size(); idx++) {
    	int i = idx / system_sizes.size();
        int j = idx % temperatures.size();
        std::vector<Vec3> positions, velocities;
        velocities.resize(N, Vec3());
        positions.resize(N, Vec3());
        double system_size = system_sizes[i];
        double T_init = temperatures[j];
        std::vector<double> sweep_potentials(sweeps, 0);
        positions = cubicLattice(N, system_size);
        System atom_system(system_size, positions, velocities);
        atom_system.computePotentialEnergy();
        double potentialEnergy = atom_system.computePotentialEnergy();
        /*
        int runup = 1000, dummy = 0;
        for (int i = 0; i < runup; i++) {
        	MC_sweep(&atom_system, &potentialEnergy, &random, dr, T_init, &dummy);
        }
        for (int i = 0; i < sweeps; i++) {
        	int Naccept = 0;
            MC_sweep(&atom_system, &potentialEnergy, &random, dr, T_init, &Naccept);
            sweep_potentials[i] = potentialEnergy;
        }
        */
        int Naccept = 0;
        MC_sweep(&atom_system, &potentialEnergy, &random, dr, T_init, &Naccept);
        double sum = std::accumulate(sweep_potentials.begin(), sweep_potentials.end(), 0.0);
        double mean = sum / sweeps;
        double accum = 0.0;
        std::for_each (sweep_potentials.begin(), sweep_potentials.end(), [&](const double d) {
        	accum += (d - mean) * (d - mean);
        });
        potentialEnergies[i*system_sizes.size() + j] = potentialEnergy;
        stdDeviations[i*system_sizes.size() + j] = std::sqrt(accum / (sweeps-1));
    }
    std::ofstream file("potentialEnergies.txt");
    for (double i : system_sizes) {
        file << i << " ";
    }
    file << "\n";
    for (double i : temperatures) {
        file << i << " ";
    }
    file << "\n";
    for (int i = 0; i < system_sizes.size(); i++) {
        for (int j = 0; j < temperatures.size(); j++) {
            file << i << " " << j << " " << potentialEnergies[i*system_sizes.size() + j] << " " << stdDeviations[i*system_sizes.size() + j] << "\n";
        }
    }
    file.close();
    return 0;
}