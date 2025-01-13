#include "verlet.hpp"
#include <chrono>
#include <fstream>

std::array<float, 100> polarangles, azimuthalangles;
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

std::vector<Vec3> randomDist(int N, float system_size,UniformRandomFloat *rd_ptr) {
    if (N <= 0) {
        throw std::invalid_argument("Number of atoms (N) must be greater than 0.");
    }
    if (system_size <= 0) {
        throw std::invalid_argument("System size must be greater than 0.");
    }
    UniformRandomFloat &rd = *rd_ptr;
    std::vector<Vec3> positions(N, Vec3());
    for (int i = 0; i < N; i++) {
        const float x = system_size * rd();
        const float y = system_size * rd();
        const float z = system_size * rd();
        positions[i] = Vec3(x, y, z);
    }
    return positions;
}

std::vector<Vec3> cubicLattice(int N, float system_size, UniformRandomFloat *rd_ptr) {
    std::vector<Vec3> positions(N, Vec3());
	UniformRandomFloat &rd = *rd_ptr;
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
    float lattice_spacing = system_size / (cube_root);
    if (lattice_spacing <= 0) {
        throw std::runtime_error("Lattice spacing must be greater than 0.");
    }

    // Calculate the starting position for the lattice
    float center = system_size / 2.0;
    float start_position = center - (lattice_spacing * (cube_root - 1) / 2);

    // Generate positions on a cubic lattice
    int count = 0;
    for (int i = 0; i < cube_root && count < N; ++i) {
        for (int j = 0; j < cube_root && count < N; ++j) {
            for (int k = 0; k < cube_root && count < N; ++k) {
                float x = start_position + (i + (1-2*rd())/4) * lattice_spacing;
                float y = start_position + (j + (1-2*rd())/4) * lattice_spacing;
                float z = start_position + (k + (1-2*rd())/4) * lattice_spacing;
                positions[count] = Vec3(x, y, z);
                count++;
            }
        }
    }
    return positions;
}


std::tuple<float, std::vector<float>> acceptanceRate(const System &atom_system, const float &sum_of_potentials, const int &atom_idx, const Vec3 &new_position, const float &T) {
    const int N = atom_system.getN();
    const float system_size = atom_system.getSystemSize();
    const std::vector<float> &potentials = atom_system.getPotentialEnergies();
    std::vector<float> new_potentials = potentials, dpotentials(N, 0);
    int cell_index = atom_system.getCell(new_position);
    float sum_of_new_potentials = sum_of_potentials;
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
    const float acceptance = std::exp(-(sum_of_new_potentials - sum_of_potentials) / (T));
    std::pair<float, std::vector<float>> result(acceptance, dpotentials);
    return result;
}

void MC_step(System *atom_system_ptr, float *sum_of_potentials_ptr, const int &atom_idx, UniformRandomFloat *rd_ptr, const float &dr, const float &T
             , int *Naccept_ptr) {
    System &atom_system = *atom_system_ptr;
    UniformRandomFloat &rd = *rd_ptr;
    float &sum_of_potentials = *sum_of_potentials_ptr;
    const std::vector<float> &potential = atom_system.getPotentialEnergies();
    const float system_size = atom_system.getSystemSize();
    const Vec3 position = atom_system.getAtom(atom_idx).getPosition();
    //const int random_index = std::floor(10000 * rd());
    //const Vec3 velocity = dr * vs[random_index];
    //const Vec3 prop_position = atom_system.PeriodicPositionUpdate(position, velocity, 1.0);
    const Vec3 displacement = dr/std::sqrt(3) * Vec3(1-2*rd(), 1-2*rd(), 1-2*rd());
    const Vec3 prop_position = atom_system.PeriodicPositionUpdate(position, displacement, 1.0);
    float acceptance_rate;
    std::vector<float> dpotentials;
    tie(acceptance_rate, dpotentials) = acceptanceRate(atom_system, sum_of_potentials, atom_idx, prop_position, T);
    int &Naccept = *Naccept_ptr;
    if (rd() < acceptance_rate) {
        atom_system.updatePosition(atom_idx, prop_position);
        std::vector<float> new_potentials = potential;
        for (int i = 0; i < atom_system.getN(); i++) {
            new_potentials[i] += dpotentials[i];
            sum_of_potentials += dpotentials[i];
        }
        atom_system.updatePotentialEnergies(new_potentials);
        Naccept++;
    }
}

void MC_sweep(System *atom_system_ptr, float *sum_of_potentials_ptr, UniformRandomFloat *rd_ptr, const float &dr
              , const float &T, int *Naccept_ptr) {
    System &atom_system = *atom_system_ptr;
    const int N = atom_system.getN();
    for (int i = 0; i < N; i++) {
        MC_step(&atom_system, sum_of_potentials_ptr, i, rd_ptr, dr, T, Naccept_ptr);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cout << "Usage: " << argv[0] << " [system_size] [N] [T_init] [sweeps] [dr] [seed]" << std::endl;
        return -1;
    }
    // parameters
    float system_size = atof(argv[1]);
    int N = atoi(argv[2]);
    float T_init = atof(argv[3])/Epsilon;
    int sweeps = atoi(argv[4]);
    float dr = atof(argv[5]);
    int seed = atoi(argv[6]);
    UniformRandomFloat random(seed);
    std::vector<Vec3> positions(N, Vec3()), velocities(N, Vec3());
    //positions = randomDist(N, system_size, &random);
    positions = cubicLattice(N, system_size, &random);
    System atom_system(system_size, positions, velocities, T_init);
    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    atom_system.computePotentialEnergy();
    float potentialEnergies = atom_system.computePotentialEnergy();
    std::ofstream file("MCdata.txt");
    file << system_size << "\n" <<  N <<  "\n" << sweeps << "\n" << seed << "\n";
	/*
    float dummy_potential = 0;
    int runup = 15000, dummy = 0;
    for (int i = 0; i < runup; i++) {
        MC_sweep(&atom_system, &dummy_potential, &random, dr, T_init, &dummy);
    }
	 */
    int Naccept = 0;
    for (int i = 0; i < sweeps; i++) {
        MC_sweep(&atom_system, &potentialEnergies, &random, dr, T_init, &Naccept);
        if (i % (sweeps/50) == 0) {
            int barWidth = 50;
            std::cout << "[";
            int pos = barWidth * i / sweeps;
            for (int j = 0; j < barWidth; ++j) {
                if (j < pos) std::cout << "=";
                else if (j == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << 2*int(i * 50 / sweeps) << " % " << static_cast<float>(Naccept)/(N*i) << "\r";
            std::cout.flush();
        }
        /*
        for (int j = 0; j < N; j++) {
            Vec3 position = atom_system.getAtom(j).getPosition();
            file << position.getX() << " " << position.getY() << " " << position.getZ() << "\n";
        }
        */
        file << potentialEnergies << " " << 0 << std::endl;
    }
    std::cout << "[" << std::string(50, '=') << "] 100%\n";
    file.close();
    // stop the timer
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed_seconds = stop-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Energy Data written to MCdata.txt\n";
    // write the final positions to a file
    std::ofstream initial_positions("initial_positions.txt");
    for (int i = 0; i < N; i++) {
        Vec3 position = atom_system.getAtom(i).getPosition();
        initial_positions << position.getX() << " " << position.getY() << " " << position.getZ() << "\n";
    }
    initial_positions.close();
    std::cout << "Initial positions written to initial_positions.txt\n";
    return 0;
}