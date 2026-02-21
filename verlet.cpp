#include "verlet.hpp"
#define SYSTEM_SIZE 25
#define NUM_ATOMS 300
#define BOX_N ((10 * SYSTEM_SIZE / 25) - 1)
#define T_INIT 10
#define STEPS 100000

#define DT 1
#define RESOLUTION 100

using json = nlohmann::json;

std::vector<Vec3> cubicLattice(const int N, const double system_size) {
    std::vector<Vec3> positions;
    positions.reserve(N);

    if (N <= 0) {
        throw std::invalid_argument("Number of atoms (N) must be greater than 0.");
    }
    if (system_size <= 0) {
        throw std::invalid_argument("System size must be greater than 0.");
    }

    // Find the next cube root greater than or equal to N
    const int cube_root = static_cast<int>(std::ceil(std::cbrt(N)));

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
        double x = start_position + i * lattice_spacing;
        for (int j = 0; j < cube_root && count < N; ++j) {
            double y = start_position + j * lattice_spacing;
            for (int k = 0; k < cube_root && count < N; ++k) {
                double z = start_position + k * lattice_spacing;
                positions.emplace_back(x, y, z);
                if (++count == NUM_ATOMS) break;
            }
            if (count == NUM_ATOMS) break;
        }
        if (count == NUM_ATOMS) break;
    }
    return positions;
}


int main(int argc, char *argv[]) {
    // parameters
    constexpr double system_size = SYSTEM_SIZE;
    constexpr size_t num_atoms = NUM_ATOMS;
    const std::vector<Vec3> positions = cubicLattice(num_atoms, system_size);
    std::vector velocities(num_atoms, Vec3());
    constexpr double T_init = T_INIT/Epsilon;
    const double sigma_v = std::sqrt(T_init/Mass);
    NormalRandomFloat random(sigma_v);
    for (int i = 0; i < num_atoms; ++i) {
        velocities[i] = {random(), random(), random()};
    }
    const char filename[]= "data.txt";
    System<BOX_N,NUM_ATOMS> atom_system(system_size, positions, velocities, T_init);
    // start time measurement
    const auto start = std::chrono::high_resolution_clock::now();
    // run the simulation
    constexpr size_t steps = STEPS;
    const double dt =
        DT * fs /
        std::sqrt(Mass * Dalton * Sigma * Sigma * nm * nm / (Epsilon * kB));
    constexpr size_t resolution = RESOLUTION;
     atom_system.run(steps, dt, &filename[0], resolution);
	std::cout << "Potential energy: " << atom_system.getPotentialEnergy() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Data written to data.txt\n";
    return 0;
}


