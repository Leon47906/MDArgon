#include "verlet.hpp"
#define NUM_ATOMS 500
#define SYSTEM_SIZE 25
#define BOX_N ((10 * SYSTEM_SIZE / 25) - 1)
#define T_INIT 50
#define SWEEPS 100000
#define RUNUP 5000
#define DR 1

using json = nlohmann::json;

std::array<double, 100> polarangles;
std::array<double, 200> azimuthalangles;
std::array<Vec3, 10000> vs;

void initialize_vs() {
    for (int i = 0; i < 100; i++) {
        polarangles[i] = M_PI * i / 100;
    }
    for (int i = 0; i < 200; i++) {
        azimuthalangles[i] = 2 * M_PI * i / 200;
    }
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 200; j++) {
            vs[i * 100 + j] = Vec3(std::sin(polarangles[i]) * std::cos(azimuthalangles[j]),
                                   std::sin(polarangles[i]) * std::sin(azimuthalangles[j]),
                                   std::cos(polarangles[i]));
        }
    }
}

std::vector<Vec3> randomDist(const int &N, const double &system_size,
    UniformRandomFloat *rd_ptr) {
    if (N <= 0) {
        throw std::invalid_argument("Number of atoms (N) must be greater than 0.");
    }
    if (system_size <= 0) {
        throw std::invalid_argument("System size must be greater than 0.");
    }
    UniformRandomFloat &rd = *rd_ptr;
    std::vector<Vec3> positions(N, Vec3());
    for (int i = 0; i < N; i++) {
        const double x = system_size * rd();
        const double y = system_size * rd();
        const double z = system_size * rd();
        positions[i] = Vec3(x, y, z);
    }
    return positions;
}

std::vector<Vec3> cubicLattice(const int N, const double system_size) {
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
                positions[count] = Vec3(x, y, z);
                count++;
            }
        }
    }
    return positions;
}

// Funktion, welche die Akzeptanzrate berechnet

void acceptanceRate(double* acceptance_ptr, double* dpotentials_ptr,
    const System<BOX_N,NUM_ATOMS> &atom_system, const double &sum_of_potentials,
    const int &atom_idx, const Vec3 &new_position, const double &T) {
    const double system_size = atom_system.getSystemSize();
    const std::array<double, NUM_ATOMS> &potentials = atom_system.getPotentialEnergies();
    std::array<double, NUM_ATOMS> new_potentials = potentials;
    double* dpotentials = dpotentials_ptr;
    double sum_of_new_potentials = sum_of_potentials;
    for (const size_t i : atom_system.getAdjacentAtoms(new_position)) {
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
    for (const size_t i : atom_system.getAdjacentAtoms(atom_idx)) {
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

// Funktion, welche einen Monte-Carlo-Schritt durchführt

void MC_step(System<BOX_N,NUM_ATOMS> *atom_system_ptr, double *sum_of_potentials_ptr,
    const int &atom_idx, UniformRandomFloat *rd_ptr, const double &dr,
    const double &T, int *Naccept_ptr) {
    auto &atom_system = *atom_system_ptr;
    UniformRandomFloat &rd = *rd_ptr;
    double &sum_of_potentials = *sum_of_potentials_ptr;
    constexpr size_t N = NUM_ATOMS;
    const std::array<double, N> &potential = atom_system.getPotentialEnergies();
    const Vec3 position = atom_system.getAtom(atom_idx).getPosition();
    const Vec3 displacement = dr/std::sqrt(3) * Vec3(2*rd()-1, 2*rd()-1, 2*rd()-1);
    const Vec3 prop_position = atom_system.PeriodicPositionUpdate(position, displacement, 1.0);
    double acceptance_rate;
    auto* dpotentials = new double[N];
    memset(dpotentials, 0, N*sizeof(double));
    acceptanceRate(&acceptance_rate, dpotentials ,atom_system, sum_of_potentials, atom_idx, prop_position, T);
    int &Naccept = *Naccept_ptr;
    if (rd() < acceptance_rate) {
        atom_system.updatePosition(atom_idx, prop_position);
        std::array<double, N> new_potentials = potential;
        for (int i = 0; i < N; i++) {
            new_potentials[i] += dpotentials[i];
            sum_of_potentials += dpotentials[i];
        }
        atom_system.updatePotentialEnergies(new_potentials);
        Naccept++;
    }
    delete[] dpotentials;
}

// Funktion, welche einen Monte-Carlo-Sweep durchführt

void MC_sweep(System<BOX_N,NUM_ATOMS> *atom_system_ptr, double *sum_of_potentials_ptr,
    UniformRandomFloat *rd_ptr, const double &dr, const double &T,
    int *Naccept_ptr) {
    auto &atom_system = *atom_system_ptr;
  constexpr size_t N = NUM_ATOMS;
    for (int i = 0; i < N; i++) {
        MC_step(&atom_system, sum_of_potentials_ptr, i, rd_ptr, dr, T, Naccept_ptr);
    }
}

// Hauptprogramm

int main(int argc, char *argv[]) {
    // parameters
    constexpr double system_size = SYSTEM_SIZE;
    constexpr size_t N = NUM_ATOMS;
    constexpr double T_init = T_INIT;
    constexpr size_t sweeps = SWEEPS;
    double dr = DR;
    constexpr size_t runup = RUNUP;
    UniformRandomFloat random{};
    std::vector<Vec3> positions(N, Vec3()), velocities(N, Vec3());
    positions = cubicLattice(N, system_size);
    initialize_vs();
    System<BOX_N,NUM_ATOMS> atom_system(system_size, positions, velocities, T_init);
    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    double potentialEnergies = atom_system.computePotentialEnergy();
    std::cout << "Initial potential energy: " << potentialEnergies << std::endl;
    std::ofstream file("MCdata.txt");
    file << system_size << "\n" << T_init << "\n" <<  N <<  "\n" << sweeps << "\n" << 1 << "\n" << 1 << "\n";
    for (int i = 0; i < runup; i++) {
    	int Naccept = 0;
        MC_sweep(&atom_system, &potentialEnergies, &random, dr, T_init, &Naccept);
        // automatische Steuerung von dr
        if (double acceptance_rate = static_cast<double>(Naccept) / (N);
            acceptance_rate < 0.15 && dr*0.9 > 0.1) {
        	dr *= 0.9;
        }
        else if (acceptance_rate > 0.25 && dr < system_size/2) {
        	dr *= 1.1;
        }
        if (i % (runup/50) == 0) {
            int barWidth = 50;
            std::cout << "[";
            size_t pos = barWidth * i / runup;
            for (int j = 0; j < barWidth; ++j) {
                if (j < pos) std::cout << "=";
                else if (j == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << 2* static_cast<int>(i * 50 / runup) << " %\r";
            std::cout.flush();
        }
    }
    std::cout << "[" << std::string(50, '=') << "] 100%\n";
    int global_Naccept = 0;
    for (size_t i = 0; i < sweeps; i++) {
      	int Naccept = 0;
        MC_sweep(&atom_system, &potentialEnergies, &random, dr, T_init, &Naccept);
        double acceptance_rate = static_cast<double>(Naccept)/(N);
        global_Naccept += Naccept;
        // adjusst dr, such that an acceptance rate of 20% is achieved
        if (acceptance_rate < 0.15 && dr*0.9 > 0.1) {
        	dr *= 0.9;
        }
        else if (acceptance_rate > 0.25 && dr*1.1 < 2.5) {
        	dr *= 1.1;
        }
        if (i % (sweeps/50) == 0) {
            int barWidth = 50;
            std::cout << "[";
            size_t pos = barWidth * i / sweeps;
            for (size_t j = 0; j < barWidth; ++j) {
                if (j < pos) std::cout << "=";
                else if (j == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << 2* (i * 50 / sweeps) << " % " << static_cast<double>(global_Naccept)/(N*(i+1)) << " " << dr << "\r";
            std::cout.flush();
        }
        file << potentialEnergies << std::endl;
    }
    std::cout << "[" << std::string(50, '=') << "] 100%\n";
    file.close();
    // stop the timer
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = stop-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Energy and Position Data written to MCdata.txt\n";
    std::cout << "Final potential energy: " << potentialEnergies << std::endl;
    return 0;
}