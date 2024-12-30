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

std::vector<Vec3> cubicLattice(int N, double system_size,UniformRandomDouble *rd_ptr) {
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

double acceptanceRate(const System &atom_system, const double &potential, const int &atom_idx, const Vec3 &new_position, const double &T) {
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
    double new_potential = atom_system.computePotentialSansAtomidx(atom_idx);
    new_potential += std::accumulate(energies.begin(), energies.end(), 0.0);
    const double acceptance = std::exp(-(new_potential - potential) / (kB * T));
    return acceptance;
}

void MC_step(System *atom_system_ptr, const double &potential, const int &atom_idx, UniformRandomDouble *rd_ptr, const double &v, const double &dt, const double &T
             , std::vector<double> *rates_ptr) {
    System &atom_system = *atom_system_ptr;
    UniformRandomDouble &rd = *rd_ptr;
    const double system_size = atom_system.getSystemSize();
    const Vec3 position = atom_system.getAtom(atom_idx).getPosition();
    const int random_index = std::floor(10000 * rd());
    const Vec3 velocity = rd() * v * vs[random_index];
    const Vec3 new_position = atom_system.PeriodicPositionUpdate(position, velocity, dt);
    const double acceptance_rate = acceptanceRate(atom_system, potential, atom_idx, new_position, T);
    if (rd() < acceptance_rate) {
        atom_system.updatePosition(atom_idx, new_position);
    }
    rates_ptr->push_back(std::min(1.0, acceptance_rate));
}

void MC_sweep(System *atom_system_ptr, UniformRandomDouble *rd_ptr, const double &v, const double &dt, const double &T
              , std::vector<double> *rates_ptr) {
    System &atom_system = *atom_system_ptr;
    const int N = atom_system.getN();
    const double potential = atom_system.getPotentialEnergy();
    for (int i = 0; i < N; i++) {
        MC_step(&atom_system, potential, i, rd_ptr, v, dt, T, rates_ptr);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << " [system_size] [N] [T_init] [sweeps] [dt]" << std::endl;
        return -1;
    }
    // parameters
    UniformRandomDouble random;
    std::vector<Vec3> positions, velocities;
    double system_size = atof(argv[1])*nm;
    int N = atoi(argv[2]);
    double T_init = atof(argv[3]);
    int sweeps = atoi(argv[4]);
    double dt = atof(argv[5])*fs;
    std::vector<double> rates;
    initialize_vs();
    positions.resize(N, Vec3());
    velocities.resize(N, Vec3());
    positions = cubicLattice(N, system_size, &random);
    double v = std::sqrt(2 * kB * T_init / Mass);
    velocities.resize(N, Vec3());
    System atom_system(system_size, positions, velocities);
    // start the timer
    auto start = std::chrono::high_resolution_clock::now();
    atom_system.computePotentialEnergy();
    std::ofstream file("MCdata.txt");
    file << system_size << "\n" <<  N <<  "\n" << sweeps << "\n";
    for (int i = 0; i < sweeps; i++) {
        MC_sweep(&atom_system, &random, v, dt, T_init, &rates);
        if (i % (sweeps/50) == 0) {
            int barWidth = 50;
            std::cout << "[";
            int pos = barWidth * i / sweeps;
            for (int j = 0; j < barWidth; ++j) {
                if (j < pos) std::cout << "=";
                else if (j == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << 2*int(i * 50 / sweeps) << " % " << std::accumulate(rates.begin(), rates.end(), 0.0)/rates.size() << "\r";
            std::cout.flush();
        }
        std::vector<Vec3> data = atom_system.getData();
        for (int j = 0; j < N; j++) {
            file << data[j].getX() << " " << data[j].getY() << " " << data[j].getZ() << "\n";
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