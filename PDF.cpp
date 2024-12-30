#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <tuple> // For std::tuple

// Function to calculate pairs in 3D and normalize by the number of particles
double calculateNormalizedPairDensity(const std::vector<std::tuple<double, double, double>>& particles) {
    // Validate inputs
    if (particles.size() < 2) {
        std::cerr << "Invalid particle data: At least two particles are required.\n";
        return 0.0;
    }

    // Calculate pair distances and their sum
    double totalDistance = 0.0;
    int pairCount = 0;

    for (size_t i = 0; i < particles.size(); ++i) {
        for (size_t j = 1; j < particles.size(); ++j) {
            if (i != j) {
                double dx = std::get<0>(particles[i]) - std::get<0>(particles[j]);
                double dy = std::get<1>(particles[i]) - std::get<1>(particles[j]);
                double dz = std::get<2>(particles[i]) - std::get<2>(particles[j]);
                double distance = dx * dx + dy * dy + dz * dz;
                if 
                ++pairCount;
            }
        }
    }

    // Return the normalized pair density
    return pairCount > 0 ? totalDistance / pairCount : 0.0;
}

int main() {
    // Read particle positions from a file
    std::ifstream inputFile("data.txt");
    if (!inputFile) {
        std::cerr << "Error: Could not open file.\n";
        return 1;
    }

    std::vector<std::tuple<double, double, double>> particles;
    std::string line;
    int get_infos = 3;
    int system_size, N, steps;

    // Skip the first three lines
    for (int i = 0; i < get_infos; ++i) {
        if (!std::getline(inputFile, line)) {
            std::cerr << "Error: File does not contain enough lines to skip.\n";
            return 1;
        }
        else {
            if (i == 0) {
                system_size = std::stof(line);
            }
            else if (i == 1) {
                N = std::stoi(line);
            }
            else if (i == 2) {
                steps = std::stoi(line);
            }
        }
    }
    std::cout << system_size << " " << N << " " << steps << std::endl;

    // Parse the file
    for (int i = 0; i < N*steps; ++i) {
        std::getline(inputFile, line);
        std::istringstream iss(line);
        double x, y, z;
        if (iss >> x >> y >> z) {
            particles.emplace_back(x, y, z);
        } else {
            std::cerr << "Error: Invalid particle data in file.\n";
            return 1;
        }
    }

    inputFile.close();

    // Calculate normalized pair density
    double result = calculateNormalizedPairDensity(particles);

    // Print the result
    std::cout << "Normalized pair density: " << result << std::endl;

    return 0;
}
