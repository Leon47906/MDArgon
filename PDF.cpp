#include <iostream>
#include <vector>
#include <tuple>
#include <array>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

// Function to calculate pairs in 3D and normalize by the number of particles
#include <thread>
#include <mutex>

// Function to calculate pairs in 3D with multithreading support
double calculateNormalizedPairDensity(
    const std::vector<std::vector<std::tuple<double, double, double>>>& particles,
    double radius, double ringsize, size_t steps, size_t N, double system_size) {

    double radiusSquared = radius * radius;
    double maxRadiusSquared = (radius + ringsize) * (radius + ringsize);
    std::vector<double> pairList;

    size_t num_repeats = std::ceil(radius / system_size); // Number of box repetitions in each dimension
    size_t num_threads = std::thread::hardware_concurrency(); // Number of available threads
    std::mutex mtx; // Mutex for thread-safe access to shared resources

    for (size_t step = 0; step < steps; ++step) {
        size_t totalPairCount = 0;

        // Lambda function for thread work
        auto worker = [&](size_t start, size_t end) {
            size_t threadPairCount = 0;
            for (size_t i = start; i < end; ++i) {
                for (size_t j = i + 1; j < N; ++j) {
                    // Compute the minimum distance considering periodic boundaries
                    double dx = std::get<0>(particles[step][i]) - std::get<0>(particles[step][j]);
                    double dy = std::get<1>(particles[step][i]) - std::get<1>(particles[step][j]);
                    double dz = std::get<2>(particles[step][i]) - std::get<2>(particles[step][j]);

                    // Check all periodic replicas
                    for (int x_shift = -num_repeats; x_shift <= num_repeats; ++x_shift) {
                        for (int y_shift = -num_repeats; y_shift <= num_repeats; ++y_shift) {
                            for (int z_shift = -num_repeats; z_shift <= num_repeats; ++z_shift) {
                                double shifted_dx = dx + x_shift * system_size;
                                double shifted_dy = dy + y_shift * system_size;
                                double shifted_dz = dz + z_shift * system_size;
                                double distanceSquared = shifted_dx * shifted_dx +
                                                         shifted_dy * shifted_dy +
                                                         shifted_dz * shifted_dz;

                                // Count pairs within the radius range
                                if (radiusSquared < distanceSquared && distanceSquared < maxRadiusSquared) {
                                    ++threadPairCount;
                                }
                            }
                        }
                    }
                }
            }

            // Safely update the total pair count
            std::lock_guard<std::mutex> lock(mtx);
            totalPairCount += threadPairCount;
        };

        // Divide work among threads
        std::vector<std::thread> threads;
        size_t chunkSize = N / num_threads;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * chunkSize;
            size_t end = (t == num_threads - 1) ? N : start + chunkSize;
            threads.emplace_back(worker, start, end);
        }

        // Wait for all threads to finish
        for (auto& thread : threads) {
            thread.join();
        }

        // Normalize the pair count and add to the pair list
        pairList.push_back(2.0 * static_cast<double>(totalPairCount) / N);
    }

    // Calculate the average normalized pair density
    double average = 0.0;
    for (double pair : pairList) {
        average += pair;
    }
    return average / pairList.size();
}


int main(int argc, char* argv[]) {
    // Read the ring size
    double ringsize = 1e-9;
    if (argc == 2) {
        ringsize = std::stod(argv[1]);
    }

    // Open the input file
    std::ifstream inputFile("data.txt");
    if (!inputFile) {
        std::cerr << "Error: Could not open file.\n";
        return 1;
    }

    // Parse header lines
    size_t system_size, N, steps, res;
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
    std::vector<std::vector<std::tuple<double, double, double>>> particles(steps, std::vector<std::tuple<double, double, double>>(N));
    for (size_t step = 0; step < steps; ++step) {
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
    std::cout << "[";
    std::vector<double> results;
    double system_limit = system_size;
    std::cout << int(ringsize/ringsize);
    for (double radius = 0.0; radius < system_limit; radius += ringsize) {
        double result = calculateNormalizedPairDensity(particles, radius, ringsize, steps, N, system_size);
        results.push_back(result);
        if (int(radius/ringsize)== 0) {
            std::cout << "=";
        }
    }
    std::cout << "] 100%\n";

    // Write results to a file
    std::ofstream outputFile("pair_results.txt");
    if (!outputFile) {
        std::cerr << "Error: Could not open output file.\n";
        return 1;
    }
    for (double result : results) {
        outputFile << std::fixed << std::setprecision(6) << result << "\n";
    }
    outputFile.close();

    std::cout << "Data written to pair_results.txt\n";
    return 0;
}
