#include "verlet.hpp"
#include <fstream>
#include <chrono>
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



// verlet algorithm to solve the system
int main(int argc, char *argv[]) {
    // parameters
    double system_size, T_init, dt;
    int N, steps;
    std::vector<Vec3> positions, velocities;
    if (argc == 6) {
        system_size = atof(argv[1])*nm;
        N = atoi(argv[2]);
        T_init = atof(argv[3]);
        steps = atoi(argv[4]);
        dt = atof(argv[5])*fs;
        positions.resize(N, Vec3());
        velocities.resize(N, Vec3());
        positions = cubicLattice(N, system_size);
        double v = std::sqrt(2 * kB * T_init / Mass);
        UniformRandomDouble random;
        for (int i = 0; i < N; i++) {
            velocities[i] = unit_velocities[std::floor(6*random())]*v;
        }
    }
    else if (argc == 1) {
        system_size = 2*nm;
        N = 7;
        steps = 75000;
        dt = 1*fs;
        /*
        positions = np.array([[0.00, 0.00], [0.02, 0.39], [0.34, 0.17], [0.36, -0.21],
                      [-0.02, -0.40], [-0.35, -0.16], [-0.31, 0.21]]) * 1e-9
        velocities = np.array([[-30.00, -20.00], [50.00, -90.00], [-70.00, -60.00], [90.00, 40.00],
                       [80.00, 90.00], [-40.00, 100.00], [-80.00, -60.00]])
         */
        positions = {Vec3(0.00*nm, 0.00*nm, 0), Vec3(0.02*nm, 0.39*nm, 0), Vec3(0.34*nm, 0.17*nm, 0), Vec3(0.36*nm, -0.21*nm, 0),
                     Vec3(-0.02*nm, -0.40*nm, 0), Vec3(-0.35*nm, -0.16*nm, 0), Vec3(-0.31*nm, 0.21*nm, 0)};
        for (int i = 0; i < positions.size(); i++) {
             positions[i] = positions[i] + Vec3(0.5*system_size, 0.5*system_size, 0);
             std::cout << positions[i].getX() << " " << positions[i].getY() << " " << positions[i].getZ() << std::endl;
        }
        velocities = {Vec3(-30.00, -20.00, 0), Vec3(50.00, -90.00, 0), Vec3(-70.00, -60.00, 0), Vec3(90.00, 40.00, 0),
                      Vec3(80.00, 90.00, 0), Vec3(-40.00, 100.00, 0), Vec3(-80.00, -60.00, 0)};
    }
    else {
        std::cout << "Usage: " << argv[0] << " [system_size] [N] [T_init] [steps] [dt]" << std::endl;
        return -1;
    }
    System atom_system(system_size, positions, velocities);
    //start time measurement
   	auto start = std::chrono::high_resolution_clock::now();
    // run the simulation
    std::pair <std::vector<std::vector<Vec3>>,std::vector<std::array<double,2>>> sim = atom_system.run(steps, dt);
    std::vector<std::vector<Vec3>> data = sim.first;
    std::vector<std::array<double,2>> energies = sim.second;
    //end time measurement
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    if (argc == 6) {
        double V = system_size * system_size * system_size;
        double A = 0.1363;
        double B = 3.22*std::pow(10,-5);
        std::cout << "Temperature: " << T_init << "K\n";
        std::cout << "Number of atoms: " << N << "\n";
        // calculate the pressure in bar, with the van der Waals equation of state
        double p = vanDerWaalsPressure(T_init, V, N, A, B);
        std::cout << "Pressure: " << p*std::pow(10,-5) << " bar\n";
    }
    // write the data to a file, which can be visualized with an animation through python
    std::cout << "Writing to file...\n";
    std::ofstream file("data.txt");
    file << system_size << "\n" <<  N <<  "\n" << steps << "\n";
    for (int i = 0; i < steps; i++) {
        for (int j = 0; j < N; j++) {
            file << std::setprecision(6) << data[i][j].getX() << " " << data[i][j].getY() << " " << data[i][j].getZ() << "\n";
        }
    }
    for (int i = 0; i < steps; i++) {
        file << std::setprecision(6) << energies[i][0] << " " << energies[i][1] << "\n";
    }
    file.close();
    std::cout << "Data written to data.txt\n";
    return 0;
}



