#include "verlet.hpp"
#include <cstring>
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

std::vector<Vec3> cubicLattice(int N, float system_size) {
    std::vector<Vec3> positions(N, Vec3());
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
                float x = start_position + i * lattice_spacing;
                float y = start_position + j * lattice_spacing;
                float z = start_position + k * lattice_spacing;
                positions[count] = Vec3(x, y, z);
                count++;
            }
        }
    }
    return positions;
}


void acceptanceRate(float* acceptance_ptr, float* dpotentials_ptr, const System &atom_system, const float &sum_of_potentials, const int &atom_idx, const Vec3 &new_position, const float &T) {
    const int N = atom_system.getN();
    const float system_size = atom_system.getSystemSize();
    const std::vector<float> &potentials = atom_system.getPotentialEnergies();
    std::vector<float> new_potentials = potentials;
    float* dpotentials = dpotentials_ptr;
    float sum_of_new_potentials = sum_of_potentials;
    /*
    int cell_index = atom_system.getCell(new_position);
    int box_N = atom_system.getBoxN();
    if (cell_index < 0 || cell_index >= box_N * box_N * box_N) {
        std::cout << new_position.getX() << " " << new_position.getY() << " " << new_position.getZ() << std::endl;
        std::cerr << "Error: Cell index " << cell_index <<" out of bounds" << std::endl;
        exit(1);
    }
    for (int i : atom_system.getAtomsInCell(cell_index)) {
    	 if (i != atom_idx) {
            Vec3 position = atom_system.getAtom(i).getPosition();
            Vec3 diff_new = PeriodicDifference(new_position, position, system_size);
            new_potentials[i] = LennardJones(diff_new.norm2());
            dpotentials[i] = new_potentials[i] - potentials[i];
            sum_of_new_potentials += dpotentials[i];
        }
    }
    for (int i : atom_system.getAtomsInNeighboringCells(cell_index)) {
      	if (i != atom_idx) {
        	Vec3 position = atom_system.getAtom(i).getPosition();
        	Vec3 diff_new = PeriodicDifference(new_position, position, system_size);
        	new_potentials[i] = LennardJones(diff_new.norm2());
        	dpotentials[i] = new_potentials[i] - potentials[i];
        	sum_of_new_potentials += dpotentials[i];
        }
    }
     */
    for (int i : atom_system.getAdjacentAtoms(new_position)) {
        if (i != atom_idx) {
            Vec3 position = atom_system.getAtom(i).getPosition();
            Vec3 diff_new = PeriodicDifference(new_position, position, system_size);
            new_potentials[i] = LennardJones(diff_new.norm2())/2;
            dpotentials[atom_idx] += new_potentials[i];
            dpotentials[i] = new_potentials[i] - potentials[i];
            sum_of_new_potentials += dpotentials[i];
        }
    }
    dpotentials[atom_idx] -= potentials[atom_idx];
    for (int i : atom_system.getAdjacentAtoms(atom_idx)) {
    	if (i != atom_idx) {
			Vec3 position = atom_system.getAtom(i).getPosition();
        	Vec3 diff_old = PeriodicDifference(atom_system.getAtom(atom_idx).getPosition(), position, system_size);
        	new_potentials[i] = LennardJones(diff_old.norm2())/2;
        	dpotentials[i] = new_potentials[i] - potentials[i];
        	sum_of_new_potentials += dpotentials[i];
        }
    }
    *acceptance_ptr = std::exp(-(sum_of_new_potentials-sum_of_potentials) / T);
}

void MC_step(System *atom_system_ptr, float *sum_of_potentials_ptr, const int &atom_idx, UniformRandomFloat *rd_ptr, const float &dr, const float &T
             , int *Naccept_ptr) {
    System &atom_system = *atom_system_ptr;
    UniformRandomFloat &rd = *rd_ptr;
    float &sum_of_potentials = *sum_of_potentials_ptr;
    const int &N = atom_system.getN();
    const std::vector<float> &potential = atom_system.getPotentialEnergies();
    const float system_size = atom_system.getSystemSize();
    const Vec3 position = atom_system.getAtom(atom_idx).getPosition();
    const Vec3 displacement = dr/std::sqrt(3) * Vec3(2*rd()-1, 2*rd()-1, 2*rd()-1);
    const Vec3 prop_position = atom_system.PeriodicPositionUpdate(position, displacement, 1.0);
    float acceptance_rate;
    /*
    std::vector<float> dpotentials;
    dpotentials.reserve(atom_system.getN());
     */
    float* dpotentials = new float[N];
    memset(dpotentials, 0, N*sizeof(float));
    acceptanceRate(&acceptance_rate, dpotentials ,atom_system, sum_of_potentials, atom_idx, prop_position, T);
    int &Naccept = *Naccept_ptr;
    if (rd() < acceptance_rate) {
        atom_system.updatePosition(atom_idx, prop_position);
        std::vector<float> new_potentials = potential;
        for (int i = 0; i < N; i++) {
            new_potentials[i] += dpotentials[i];
            sum_of_potentials += dpotentials[i];
        }
        atom_system.updatePotentialEnergies(new_potentials);
        Naccept++;
    }
    delete[] dpotentials;
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
    if (argc != 8) {
        std::cout << "Usage: " << argv[0] << " [system_size] [N] [T_init] [sweeps] [dr] [seed] [runup]" << std::endl;
        return -1;
    }
    // parameters
    float system_size = atof(argv[1]);
    int N = atoi(argv[2]);
    float T_init = atof(argv[3])/Epsilon;
    int sweeps = atoi(argv[4]);
    float dr = atof(argv[5]);
    int seed = atoi(argv[6]);
    int runup = atoi(argv[7]);
    UniformRandomFloat random(seed);
    std::vector<Vec3> positions(N, Vec3()), velocities(N, Vec3());
    positions = cubicLattice(N, system_size);
    initialize_vs();
    System atom_system(system_size, positions, velocities, T_init);
    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    float potentialEnergies = atom_system.computePotentialEnergy();
    std::cout << "Initial potential energy: " << potentialEnergies << std::endl;
    std::ofstream file("MCdata.txt");
    file << system_size << "\n" << T_init << "\n" <<  N <<  "\n" << sweeps << "\n" << 1 << "\n" << 1 << "\n";
    for (int i = 0; i < runup; i++) {
    	int Naccept = 0;
        MC_sweep(&atom_system, &potentialEnergies, &random, dr, T_init, &Naccept);
        float acceptance_rate = static_cast<float>(Naccept)/(N);
        if (acceptance_rate < 0.15 && dr > 0.1) {
        	dr *= 0.9;
        }
        else if (acceptance_rate > 0.25 && dr < system_size/2) {
        	dr *= 1.1;
        }
        if (i % (runup/50) == 0) {
            int barWidth = 50;
            std::cout << "[";
            int pos = barWidth * i / runup;
            for (int j = 0; j < barWidth; ++j) {
                if (j < pos) std::cout << "=";
                else if (j == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << 2*int(i * 50 / runup) << " %\r";
            std::cout.flush();
        }
    }
    std::cout << "[" << std::string(50, '=') << "] 100%\n";
    int global_Naccept = 0;
    for (int i = 0; i < sweeps; i++) {
        if (potentialEnergies != potentialEnergies) {
            std::cerr << "Error: Energy is NaN" << std::endl;
            break;
        }
      	int Naccept = 0;
        MC_sweep(&atom_system, &potentialEnergies, &random, dr, T_init, &Naccept);
        float acceptance_rate = static_cast<float>(Naccept)/(N);
        global_Naccept += Naccept;
        // adjusst dr, such that an acceptance rate of 20% is achieved
        if (acceptance_rate < 0.15 && dr > 0.1) {
        	dr *= 0.9;
        }
        else if (acceptance_rate > 0.25 && dr < 2.5) {
        	dr *= 1.1;
        }
        if (i % (sweeps/50) == 0) {
            int barWidth = 50;
            std::cout << "[";
            int pos = barWidth * i / sweeps;
            for (int j = 0; j < barWidth; ++j) {
                if (j < pos) std::cout << "=";
                else if (j == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << 2*int(i * 50 / sweeps) << " % " << static_cast<float>(global_Naccept)/(N*(i+1)) << " " << dr << "\r";
            std::cout.flush();
        }
        /*
		for (int j = 0; j < N; j++) {
            Vec3 position = atom_system.getAtom(j).getPosition();
        	file << position.getX() << " " << position.getY() << " " << position.getZ() << "\n";
        }
         */
        file << potentialEnergies << std::endl;
    }
    std::cout << "[" << std::string(50, '=') << "] 100%\n";
    file.close();
    // stop the timer
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed_seconds = stop-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Energy and Position Data written to MCdata.txt\n";
    std::cout << "Final potential energy: " << potentialEnergies << std::endl;
    return 0;
}