#include "verlet.hpp"
#include <chrono>
#include <fstream>

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
    double lattice_spacing = system_size / cube_root;
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

double acceptanceRate(const System &atom_system, const int &atom_idx, const Vec3 &new_position, const double &T) {
    const int N = atom_system.getN();
    const double system_size = atom_system.getSystemSize();
    std::vector<double> energies;
    int cell_index = atom_system.getCell(new_position);
    std::vector<int> adjacent_atoms = atom_system.getAdjacentAtoms(atom_idx);
    for (int i : adjacent_atoms) {
        Vec3 position = atom_system.getAtom(i).getPosition();
        Vec3 diff = atom_system.PeriodicDifferece(new_position, position, system_size);
        energies.push_back(LennardJones(diff.norm2()));
    }
    const double old_potential = atom_system.getPotentialEnergy();
    double new_potential = atom_system.computePotentialSansAtomidx(atom_idx);
    new_potential += std::accumulate(energies.begin(), energies.end(), 0.0);
    return std::exp(-(new_potential - old_potential) / (kB * T));
}

void MC_step(System *atom_system_ptr, const int &atom_idx, UniformRandomDouble *rd_ptr, const double &v, const double &dt, const double &T) {
    System &atom_system = *atom_system_ptr;
    UniformRandomDouble &rd = *rd_ptr;
    const double system_size = atom_system.getSystemSize();
    const Vec3 position = atom_system.getAtom(atom_idx).getPosition();
    const int random_index = std::floor(10000 * rd());
    const Vec3 velocity = v * vs[random_index];
    const Vec3 new_position = atom_system.PeriodicPositionUpdate(position, velocity, dt);
    const double acceptance_rate = acceptanceRate(atom_system, atom_idx, new_position, T);
    if (rd() < acceptance_rate) {
        atom_system.updatePosition(atom_idx, new_position);
    }
}

void MC_sweep(System *atom_system_ptr, const double &v, const double &dt, const double &T) {
    System &atom_system = *atom_system_ptr;
    const int N = atom_system.getN();
    UniformRandomDouble rd;
    for (int i = 0; i < N; i++) {
        MC_step(&atom_system, i, &rd, v, dt, T);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << " [system_size] [N] [T_init] [sweeps] [dt]" << std::endl;
        return -1;
    }
    // parameters
    std::vector<Vec3> positions, velocities;
    double system_size = atof(argv[1])*nm;
    int N = atoi(argv[2]);
    double T_init = atof(argv[3]);
    int sweeps = atoi(argv[4]);
    double dt = atof(argv[5])*fs;
    initialize_vs();
    positions.resize(N, Vec3());
    velocities.resize(N, Vec3());
    positions = cubicLattice(N, system_size);
    double v = std::sqrt(2 * kB * T_init / Mass);
    UniformRandomDouble random;
    velocities.resize(N, Vec3());
    System atom_system(system_size, positions, velocities);
    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    atom_system.computePotentialEnergy();

    return 0;
}