#include "verlet.hpp"
#include <fstream>
#include <chrono>


class BoltzmannDistribution {
	std::random_device rd;
	std::mt19937 gen;
	std::normal_distribution<float> dis;
public:
	BoltzmannDistribution(float _T) : gen(rd()), dis(0.0, std::sqrt(_T)) {}
	float operator()() {
		return dis(gen);
	}
};

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



// verlet algorithm to solve the system
int main(int argc, char *argv[]) {
    // parameters
    float system_size, T_init, dt;
    int N, steps, resolution;
    std::vector<Vec3> positions, velocities;
    if (argc == 7) {
		// read the parameters from the command line
		// system size in Sigma
		system_size = atof(argv[1]);
		// number of atoms
        N = atoi(argv[2]);
		// initial temperature from K to reduced units
		T_init = atof(argv[3]) / Epsilon;
        steps = atoi(argv[4]);
		// time step from fs to reduced units
        dt = atof(argv[5])*fs/std::sqrt(Mass*Dalton*Sigma*Sigma*nm*nm/(Epsilon*kB));
        resolution = atof(argv[6]);
        positions.resize(N, Vec3());
        velocities.resize(N, Vec3());
        positions = cubicLattice(N, system_size);
        UniformRandomFloat random(1234);
		float v0 = std::sqrt(3 * T_init);
        for (int i = 0; i < N; i++) {
			velocities[i] = v0 * unit_velocities[std::floor(6 * random())];
        }
        // print the distance which one particle with velocity v0 travels in one time step
        std::cout << "Distance: " << v0 * dt << std::endl;
        /*
        Vec3 v_com = Vec3();
        for (int i = 0; i < N; i++) {
            v_com += velocities[i]/N;
        }
        for (int i = 0; i < N; i++) {
            velocities[i] -= v_com;
        }
         */
    }
    else if (argc == 2) {
        system_size = 4;
        N = 1;
        steps = 10000;
        dt = 1*fs/std::sqrt(Mass*Dalton*Sigma*Sigma*nm*nm/(Epsilon*kB));
        resolution = 50;
        positions = {Vec3(1,1,1)};
        velocities = {Vec3(-1,0,0)};
    }
    else if (argc == 1) {
        system_size = 2;
        N = 7;
        steps = 75000;
        dt = 1*fs;
        /*
        positions = np.array([[0.00, 0.00], [0.02, 0.39], [0.34, 0.17], [0.36, -0.21],
                      [-0.02, -0.40], [-0.35, -0.16], [-0.31, 0.21]]) * 1e-9
        velocities = np.array([[-30.00, -20.00], [50.00, -90.00], [-70.00, -60.00], [90.00, 40.00],
                       [80.00, 90.00], [-40.00, 100.00], [-80.00, -60.00]])
         */
        positions = {Vec3(0.00, 0.00, 0), Vec3(0.02, 0.39, 0), Vec3(0.34, 0.17, 0), Vec3(0.36, -0.21, 0),
                     Vec3(-0.02, -0.40, 0), Vec3(-0.35, -0.16, 0), Vec3(-0.31, 0.21, 0)};
        for (int i = 0; i < positions.size(); i++) {
             positions[i] = positions[i] + Vec3(0.5*system_size, 0.5*system_size, 0);
             std::cout << positions[i].getX() << " " << positions[i].getY() << " " << positions[i].getZ() << std::endl;
        }
        velocities = {Vec3(-30.00, -20.00, 0), Vec3(50.00, -90.00, 0), Vec3(-70.00, -60.00, 0), Vec3(90.00, 40.00, 0),
                      Vec3(80.00, 90.00, 0), Vec3(-40.00, 100.00, 0), Vec3(-80.00, -60.00, 0)};
    }
    else {
        std::cout << "Usage: " << argv[0] << " [system_size] [N] [T_init] [steps] [dt] [resolution]" << std::endl;
        return -1;
    }
    char filename[]= "data.txt";
    System atom_system(system_size, positions, velocities, T_init);
    //start time measurement
   	auto start = std::chrono::high_resolution_clock::now();
    // run the simulation
    atom_system.run(steps, dt, &filename[0], resolution);
	std::cout << "Potential energy: " << atom_system.getPotentialEnergy() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsed_seconds = end-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Data written to data.txt\n";
    return 0;
}



