#include <iostream>
#include <vector>
#include <tuple>
#include <array>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "verlet.hpp"

// Periodic boundary conditions
inline Vec3 PeriodicDifferece(const Vec3& r1,const Vec3& r2, const double& period) {
        const Vec3 r = r1 - r2;
        double x = r.getX();
        double y = r.getY();
        double z = r.getZ();
        while (x > period * 0.5) x -= period;
        while (x < -period * 0.5) x += period;

        while (y > period * 0.5) y -= period;
        while (y < -period * 0.5) y += period;

        while (z > period * 0.5) z -= period;
        while (z < -period * 0.5) z += period;
        return Vec3(x, y, z);
}

// Function to calculate pairs in 3D with multithreading support
std::vector<double> calculateNormalizedPairDensity(
    const std::vector<std::vector<std::tuple<double, double, double>>>& particles, double ringsize, size_t steps, size_t N, double system_size, double system_limit) {
    
    size_t num_rings = system_limit / ringsize;
    std::vector<double> counts(num_rings, 0);
    std::vector<std::vector<double>> norm_counts;
    int num_rec = system_limit / system_size -1;

    std::cout << "num_rings: " << num_rings << "\n";
    std::cout << "[";

    // Parallelizing over the steps
    #pragma omp parallel for
    for (size_t step = 0; step < steps; ++step) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                double x, y, z = std::get<0>(particles[step][i]);
                double x2, y2, z2 = std::get<0>(particles[step][j]);
                Vec3 dr1 = Vec3(x, y, z);
                Vec3 dr2 = Vec3(x2, y2, z2);

                // Calculate the distance between the particles for periodic boundary conditions
                /**/
                for (int i = -num_rec; i <= num_rec; ++i) {
                    for (int j = -num_rec; j <= num_rec; ++j) {
                        for (int k = -num_rec; k <= num_rec; ++k) {
                            Vec3 dr = dr1 - dr2 + Vec3(i*system_size, j*system_size, k*system_size);
                            double distanceSquared = dr.norm2();
                            double distance = std::sqrt(distanceSquared);
                            size_t ring = std::floor(distance/ringsize);
                            if (ring < num_rings) {
                                counts[ring]++;
                            }
                        }
                    }
                }
                /*
                            
                // Calculate the distance between the particles
                double distanceSquared = PeriodicDifferece(dr1, dr2, system_size).norm2();
                double distance = std::sqrt(distanceSquared);
                size_t ring = std::floor(distance/ringsize);
                if (ring < num_rings) {
                    counts[ring]++;
                }
                */
            }
        }

        // After finishing the work for this step, accumulate the local counts into global counts
        for (size_t i = 0; i < num_rings; ++i) {
            counts[i] = counts[i]/N/*(N*4/3*M_PI*(std::pow((i+1), 3) - std::pow(i, 3)))*/;
        }

        norm_counts.push_back(counts);

        // Print progress
        if (steps > 99) {
            if (step % (steps / 100) == 0) {
                std::cout << "=";
            }
        } else {
            std::cout << "=";
        }
    }

    std::cout << "]\n";

    // Calculate the average normalized pair density
    std::cout << "[";
    std::vector<double> results(num_rings, 0);
    for (size_t i = 0; i < num_rings; ++i) {
        double sum = 0;
        for (size_t j = 0; j < steps; ++j) {
            sum += norm_counts[j][i];
        }
        results[i] = sum/steps;
        if (num_rings > 99) {
            if (i % (num_rings / 100) == 0) {
                std::cout << "=";
            }
        } else {
            std::cout << "=";
        }
    }
    std::cout << "]" << "\n";
    
    return results;
}


int main(int argc, char* argv[]) {
    // Read max_ring from command line
    int max_ring;
    if (argc == 2) {
        max_ring = std::stoi(argv[1]);    
    } else {
        std::cout << "Usage: " << argv[0] << " [max_ring]\n";
        return -1;
    }
    

    // Open the input file
    std::ifstream inputFile("data.txt");
    if (!inputFile) {
        std::cerr << "Error: Could not open file.\n";
        return 1;
    }

    // Parse header lines
    size_t N, steps, res;
    double system_size;
    std::string line;
    for (int i = 0; i < 4; ++i) {
        if (!std::getline(inputFile, line)) {
            std::cerr << "Error: Missing header lines in the input file.\n";
            return 1;
        }
        std::istringstream iss(line);
        if (i == 0) {
            iss >> system_size;
        } else if (i == 1) {
            iss >> N;
        } else if (i == 2) {
            iss >> steps;
        } else if (i == 3) {
            iss >> res;
        }
    }
    std::cout << "system_size: " << system_size << ", ";
    std::cout << "N: " << N << ", ";
    std::cout << "steps: " << steps << ", ";
    std::cout << "res: " << res << "\n";

    // Read particle data
    std::vector<std::vector<std::tuple<double, double, double>>> particles(steps/res, std::vector<std::tuple<double, double, double>>(N));
    for (size_t step = 0; step < steps/res; ++step) {
        for (size_t i = 0; i <= N; ++i) {
            if (i != N) {
                if (!std::getline(inputFile, line)) {
                    std::cerr << "Error: Insufficient particle data.\n";
                    return 1;
                }
                std::istringstream iss(line);
                double x, y, z;
                if (iss >> x >> y >> z) {
                    particles[step][i] = std::make_tuple(x, y, z);
                } else {
                    std::cerr << "Error: Invalid particle data in file.\n";
                    return 1;
                }
            } else {
                std::getline(inputFile, line);
            }
        }
    }

    // Close the input file
    inputFile.close();

    // Calculate normalized pair density for all radii
    int cube_root = static_cast<int>(std::ceil(std::cbrt(N)));
    int cube_number = cube_root * cube_root * cube_root;

    // Calculate the lattice spacing
    double ringsize = system_size / (cube_root*20);

    std::cout << "ringsize: " << ringsize << ", ";
    std::vector<double> results;
    double system_limit = max_ring*system_size/2;
    std::cout << "system_limit: " << system_limit << ", ";
    results = calculateNormalizedPairDensity(particles, ringsize, steps/res, N, system_size, system_limit);
        
    // Write results to a file
    std::ofstream outputFile("pair_results.txt");
    if (!outputFile) {
        std::cerr << "Error: Could not open output file.\n";
        return 1;
    }
    for (double result : results) {
        outputFile << std::fixed << std::setprecision(9) << result << "\n";
    }
    outputFile.close();

    std::cout << "Data written to pair_results.txt\n";
    return 0;
}