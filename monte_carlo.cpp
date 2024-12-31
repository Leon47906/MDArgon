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
    const int random_index = std::floor(10000 * rd());
    const Vec3 velocity = dr * vs[random_index];
    const Vec3 prop_position = atom_system.PeriodicPositionUpdate(position, velocity, 1.0);
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

void MC_sweep(System *atom_system_ptr, double *sum_of_potentials_ptr, UniformRandomDouble *rd_ptr, const double &dr
              , const double &T, int *Naccept_ptr) {
    System &atom_system = *atom_system_ptr;
    const int N = atom_system.getN();
    for (int i = 0; i < N; i++) {
        MC_step(&atom_system, sum_of_potentials_ptr, i, rd_ptr, dr, T, Naccept_ptr);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        std::cout << "Usage: " << argv[0] << " [system_size] [N] [T_init] [sweeps] [dr] [resolution]" << std::endl;
        return -1;
    }
    // parameters
    UniformRandomDouble random;
    std::vector<Vec3> positions, velocities;
    double system_size = atof(argv[1])*nm;
    int N = atoi(argv[2]);
    double T_init = atof(argv[3]);
    int sweeps = atoi(argv[4]);
    double dr = atof(argv[5])*nm;
    int resolution = atoi(argv[6]);
    initialize_vs();
    positions.resize(N, Vec3());
    velocities.resize(N, Vec3());
    positions = randomDist(N, system_size, &random);
    velocities.resize(N, Vec3());
    System atom_system(system_size, positions, velocities);
    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    atom_system.computePotentialEnergy();
    double potentialEnergies = atom_system.computePotentialEnergy();
    std::ofstream file("MCdata.txt");
    file << system_size << "\n" <<  N <<  "\n" << sweeps << "\n" << resolution << "\n";
    int runup = 5000, dummy = 0;
    for (int i = 0; i < runup; i++) {
        MC_sweep(&atom_system, &potentialEnergies, &random, dr, T_init, &dummy);
    }
    for (int i = 0; i < sweeps; i++) {
    	int Naccept = 0;
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
            std::cout << "] " << 2*int(i * 50 / sweeps) << " % " << static_cast<float>(Naccept)/(N) << "\r";
            std::cout.flush();
        }
        if (i % resolution == 0) {
        	/*
        	std::vector<Vec3> data = atom_system.getData();
        	for (int j = 0; j < N; j++) {
        	    file << data[j].getX() << " " << data[j].getY() << " " << data[j].getZ() << "\n";
        	}
        	 */
            file << potentialEnergies << std::endl;
		}
    }
    std::cout << "[" << std::string(50, '=') << "] 100%\n";
    file.close();
    // stop the timer
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = stop-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    return 0;
}