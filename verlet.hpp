#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <numeric>
#include <math.h>
#include "fastinversesquareroot.hpp"
#include <chrono>

#ifndef VERLET_HPP
#define VERLET_HPP

constexpr float kB = 1.38064852e-23;
constexpr float nm = 1e-9;//nanometer
constexpr float ns = 1e-9;//nanosecond
constexpr float fs = 1e-15;//femtosecond
constexpr float Dalton = 1.66053906660e-27;//Dalton in kg
constexpr float Sigma = 0.33916; //Sigma in nm
constexpr float Epsilon = 137.9; //Epsilon in kB*K
constexpr float shift = -0.016316891136;
constexpr float Mass = 39.948; //Mass in Dalton
constexpr float one_over_sqrt_pi = 0.5*M_2_SQRTPI;


inline float LennardJones(float r2) {
	if (r2 > 6.25) return 0;
  	float r6 = r2*r2*r2;
    return 4.0/r6*(1.0/r6 - 1.0)-shift;
}

inline float ComputeAccel(float r2) {
    if (r2 > 6.25) return 0;
    else {
    	float r6 = r2*r2*r2;
        float r_inv = 1/std::sqrt(r2);
        return 24.0 * r_inv / Mass / r6 * (2.0/r6 - 1.0);
    }
}

class UniformRandomFloat{
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis;
    public:
    UniformRandomFloat() : gen(rd()), dis(0, 1) {}
    UniformRandomFloat(const int& seed) : gen(seed), dis(0, 1) {}
    float operator()() {
        return dis(gen);
    }
};

class Vec3{
	float x, y, z;
    public:
	Vec3() : x(0), y(0), z(0) {}
	Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
	Vec3(const Vec3& v) : x(v.x), y(v.y), z(v.z) {}
    Vec3& operator=(const Vec3& v) {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
    Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }
    Vec3& operator+=(const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }
    Vec3& operator-=(const Vec3& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    Vec3 operator*(float s) const {
        return Vec3(x * s, y * s, z * s);
    }
    friend Vec3 operator*(float s, const Vec3& v) {
        return v * s;
    }
	Vec3 operator/(float s) const {
		if (s == 0) {
			throw std::invalid_argument("Division by zero.");
		}
		return Vec3(x / s, y / s, z / s);
	}
	friend Vec3 operator/(float s, const Vec3& v) {
		return v / s;
	}
    float getX() const { return x; }
    float getY() const { return y; }
    float getZ() const { return z; }
    float norm2() const { return x*x + y*y + z*z; }
    Vec3 Zero() const { return Vec3(0, 0, 0); }
};

inline float dot(const Vec3& v1, const Vec3& v2) {
    return v1.getX() * v2.getX() + v1.getY() * v2.getY() + v1.getZ() * v2.getZ();
}

const static std::vector<Vec3> unit_velocities{Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1), Vec3(-1,0,0), Vec3(0,-1,0), Vec3(0,0,-1)};

class Atom{
	Vec3 position, velocity;
    public:
    Atom() : position(Vec3()), velocity(Vec3()) {}
    Atom(const Vec3 _position, const Vec3 _velocity) : position(_position), velocity(_velocity) {}
    Atom(const Atom& other) : position(other.position), velocity(other.velocity) {}
    Atom& operator=(const Atom& other) {
        if (this != &other) {
            position = other.position;
            velocity = other.velocity;
        }
        return *this;
    }
    // Copy assignment operator

    Vec3 getPosition() const { return position; }
    Vec3 getVelocity() const { return velocity; }

    void setPosition(const Vec3& position) { this->position = position; }
    void setVelocity(const Vec3& velocity) { this->velocity = velocity; }
};

class System{
    float system_size, N;
    int box_N=std::ceil(system_size/2.5);
    float box_L=system_size/box_N;
    std::vector<std::vector<int>> cells;
    std::vector<Atom> atoms;
    std::vector<Vec3> accels;
    std::vector<float> E_pot, E_kin;
    float T_init;
    public:
    System(float _system_size, std::vector<Vec3> _positions, std::vector<Vec3> _velocities, float _T_init) : system_size(_system_size), N(_positions.size()),
      T_init(_T_init) {
        accels.resize(N, Vec3());
        E_pot.resize(N,0);
        E_kin.resize(N,0);
        cells.resize(box_N*box_N*box_N);
        for (int i = 0; i < N; i++) {
            Vec3 position = _positions[i];
            Vec3 velocity = _velocities[i];
            atoms.push_back(Atom(position, velocity));
            int index = 0;
            index += std::floor(position.getX() / box_L);
            index += std::floor(position.getY() / box_L) * box_N;
            index += std::floor(position.getZ() / box_L) * box_N * box_N;
            cells[index].push_back(i);
        }
    }
    // assignment operator
    System& operator=(const System& other) {
        if (this != &other) {
            system_size = other.system_size;
            N = other.N;
            box_N = other.box_N;
            box_L = other.box_L;
            cells = other.cells;
            atoms = other.atoms;
            accels = other.accels;
            E_pot = other.E_pot;
            E_kin = other.E_kin;
        }
        return *this;
    }
    inline int getN() const { return N; }
    inline std::vector<Atom> getAtoms() const { return atoms; }
    inline int getBoxN() const { return box_N; }
    inline std::vector<std::vector<int>> getCells() const { return cells; }
    inline float getSystemSize() const { return system_size; }
    inline float getPotentialEnergy() const {
        return std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
    }
    inline std::vector<float> getPotentialEnergies() const { return E_pot; }
    inline void updatePotentialEnergies(const std::vector<float>& new_potentials) {
        	E_pot = new_potentials;
    }
    // I want to implement a function, that give me the indices of the neighboring cells
    inline int getCell(Vec3 position) const {
        //Vec3 position = atoms[atom_index].getPosition();
        int index = 0;
        index += std::floor(position.getX() / box_L);
        index += std::floor(position.getY() / box_L) * box_N;
        index += std::floor(position.getZ() / box_L) * box_N * box_N;
        return index;
    }
    inline Atom getAtom(int atom_index) const {
        return atoms[atom_index];
    }
    inline std::vector<int> getNeighboringCells(int cell_index) const {
        std::vector<int> neighbors;
        int x = cell_index % box_N;
        int y = (cell_index / box_N) % box_N;
        int z = (cell_index / box_N / box_N) % box_N;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    int x_neighbor = (x + i + box_N) % box_N;
                    int y_neighbor = (y + j + box_N) % box_N;
                    int z_neighbor = (z + k + box_N) % box_N;
                    int neighbor_index = x_neighbor + y_neighbor * box_N + z_neighbor * box_N * box_N;
                    if (cell_index!=neighbor_index) neighbors.push_back(neighbor_index);
                }
            }
        }
        //return the sorted list of neighbors
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        return neighbors;
    }
    inline std::vector<int> getAtomsInCell(int cell_index) const {
        return cells[cell_index];
    }
    inline std::vector<int> getAtomsInNeighboringCells(int cell_index) const {
        std::vector<int> neighbors = getNeighboringCells(cell_index);
        std::vector<int> atoms_in_neighbors;
        for (int neighbor : neighbors) {
            std::vector<int> atoms_in_cell = getAtomsInCell(neighbor);
            atoms_in_neighbors.insert(atoms_in_neighbors.end(), atoms_in_cell.begin(), atoms_in_cell.end());
        }
        return atoms_in_neighbors;
    }
    inline std::vector<int> getAdjacentAtoms(int atom_index) const{
		int cell_index = getCell(atoms[atom_index].getPosition());
    	std::vector<int> adjacent_atoms;
        for (int atom : getAtomsInCell(cell_index)) {
            if (atom != atom_index) {
                adjacent_atoms.push_back(atom);
            }
        }
        std::vector<int> neighbors = getAtomsInNeighboringCells(cell_index);
        adjacent_atoms.insert(adjacent_atoms.end(),neighbors.begin(),neighbors.end());
        return neighbors;
    }
	void show_neighboring_cells(int cell_index){
		std::vector<int> neighbors = getNeighboringCells(cell_index);
        std::cout << "Neighboring cells of cell " << cell_index << " are: ";
        for (int neighbor : neighbors) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
	}
    void show_adjacent_atoms(int atom_index) {
		std::vector<int> adjacent_atoms = getAdjacentAtoms(atom_index);
        std::cout << "Adjacent atoms of atom " << atom_index << " are: ";
        for (int atom : adjacent_atoms) {
            std::cout << atom << " ";
        }
        std::cout << std::endl;
    }
    void display() {
        for (int i = 0; i < N; i++) {
            std::cout << "Atom " << i << " position: " << atoms[i].getPosition().getX() << " "
                      << atoms[i].getPosition().getY() << " " << atoms[i].getPosition().getZ() << std::endl;
            std::cout << "Atom " << i << " velocity: " << atoms[i].getVelocity().getX() << " "
                      << atoms[i].getVelocity().getY() << " " << atoms[i].getVelocity().getZ() << std::endl;
        }
    }
    inline Vec3 PeriodicDifferece(const Vec3& r1,const Vec3& r2, const float& period) const {
        const Vec3 r = r1 - r2;
        float x = r.getX();
        float y = r.getY();
        float z = r.getZ();
        while (x > period * 0.5) x -= period;
        while (x < -period * 0.5) x += period;

        while (y > period * 0.5) y -= period;
        while (y < -period * 0.5) y += period;

        while (z > period * 0.5) z -= period;
        while (z < -period * 0.5) z += period;
        return Vec3(x, y, z);
    }
    inline Vec3 PeriodicPositionUpdate(const Vec3& position, const Vec3& velocity, const float& dt) const {
        Vec3 new_position = position + velocity * dt;
        float x = new_position.getX();
        float y = new_position.getY();
        float z = new_position.getZ();
        while (x < 0) x += system_size;
        while (x > system_size) x -= system_size;
        while (y < 0) y += system_size;
        while (y > system_size) y -= system_size;
        while (z < 0) z += system_size;
        while (z > system_size) z -= system_size;
        return Vec3(x, y, z);
    }
	// Non-parallel version of the computeAccels function
    void computeAccels() {
      	// Reset accelerations and potential energies
        std::fill(accels.begin(), accels.end(), Vec3());
        std::fill(E_pot.begin(), E_pot.end(), 0);
        for (int cell = 0; cell < box_N * box_N * box_N; ++cell) {
            const std::vector<int>& cell_atoms = cells[cell];
            const std::vector<int>& neighboring_cells = getNeighboringCells(cell);
            if (cell_atoms.empty()) continue;
            // Compute interactions within the same cell
            for (int atom_i : cell_atoms) {
                for (int atom_j : cell_atoms) {
                  	if (atom_i < atom_j) {
                    Vec3 ri = atoms[atom_i].getPosition();
                    Vec3 rj = atoms[atom_j].getPosition();
                    Vec3 r = ri - rj;
                    float r2 = r.norm2();
                    Vec3 accel = r*ComputeAccel(r2);
                    float pot = LennardJones(r2);
                    // Lennard Jones potential
                    E_pot[atom_i] += pot;
                    E_pot[atom_j] += pot;
                    accels[atom_i] += accel;
                    accels[atom_j] -= accel; // Newton's Third Law
                    }
                }
            }
            // Compute interactions with neighboring cells
            for (int neighbor_cell : neighboring_cells) {
                const std::vector<int>& neighbor_atoms = cells[neighbor_cell];
                for (int atom_i : cell_atoms) {
                    for (int atom_j : neighbor_atoms) {
                        Vec3 r = PeriodicDifferece(atoms[atom_i].getPosition(), atoms[atom_j].getPosition(),system_size);
                        float r2 = r.norm2();
                        Vec3 accel = r*ComputeAccel(r2);
                        float pot = LennardJones(r2);
                        accels[atom_i] += accel;
                        E_pot[atom_i] += pot;
                    }
                }
            }
        }
    }
    float computePotentialEnergy() {
        std::fill(E_pot.begin(), E_pot.end(), 0);
        for (int cell = 0; cell < box_N * box_N * box_N; ++cell) {
            const std::vector<int>& cell_atoms = cells[cell];
            const std::vector<int>& neighboring_cells = getNeighboringCells(cell);
            // Compute interactions within the same cell
            for (int atom_i : cell_atoms) {
                for (int atom_j : cell_atoms) {
                  	if (atom_i < atom_j) {
                    Vec3 ri = atoms[atom_i].getPosition();
                    Vec3 rj = atoms[atom_j].getPosition();
                    Vec3 r = ri - rj;
                    float r2 = r.norm2();
                    float pot = LennardJones(r2);
                    // Lennard Jones potential
                    E_pot[atom_i] += pot;
                    E_pot[atom_j] += pot;
                    }
                }
            }
            // Compute interactions with neighboring cells
            for (int neighbor_cell : neighboring_cells) {
                const std::vector<int>& neighbor_atoms = cells[neighbor_cell];
                for (int atom_i : cell_atoms) {
                    for (int atom_j : neighbor_atoms) {
                        Vec3 r = PeriodicDifferece(atoms[atom_i].getPosition(), atoms[atom_j].getPosition(),system_size);
                        float r2 = r.norm2();
                        float pot = LennardJones(r2);
                        E_pot[atom_i] += pot;
                    }
                }
            }
        }
        return std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
    }
    float computePotentialSansAtomidx(const int &atom_idx) const{
        std::vector<float> temp_E_pot = E_pot;
        int cell = getCell(atoms[atom_idx].getPosition());
        const std::vector<int>& cell_atoms = cells[cell];
        const std::vector<int>& neighboring_cells = getNeighboringCells(cell);
        for (int atom_i : cell_atoms) {
            if (atom_i != atom_idx) {
                Vec3 ri = atoms[atom_i].getPosition();
                Vec3 r = atoms[atom_idx].getPosition() - ri;
                float r2 = r.norm2();
                float pot = LennardJones(r2);
                temp_E_pot[atom_i] -= pot;
            }
        }
        for (int neighbor_cell : neighboring_cells) {
            const std::vector<int>& neighbor_atoms = cells[neighbor_cell];
            for (int atom_i : cell_atoms) {
                for (int atom_j : neighbor_atoms) {
                    Vec3 r = PeriodicDifferece(atoms[atom_i].getPosition(), atoms[atom_j].getPosition(),system_size);
                    float r2 = r.norm2();
                    float pot = LennardJones(r2);
                    temp_E_pot[atom_i] -= pot;
                }
            }
        }
        return std::accumulate(temp_E_pot.begin(), temp_E_pot.end(), 0.0);
    }
	inline void update_positions(float dt){
		for (int i = 0; i < N; i++) {
        	Vec3 position = atoms[i].getPosition();
            Vec3 velocity = atoms[i].getVelocity();
            //transform the position according to periodic boundary conditions
            position = PeriodicPositionUpdate(position,velocity,dt);
            atoms[i].setPosition(position);
    	}
    }
    inline void update_velocities(float dt){
    	for (int i = 0; i < N; i++) {
        	Vec3 velocity = atoms[i].getVelocity();
            Vec3 accel = accels[i];
            velocity = velocity + accel * dt;
            atoms[i].setVelocity(velocity);
            float v2 = velocity.norm2();
			//kinetic energy in kB*K
            E_kin[i] = 0.5 * Mass * v2;
        }
    }
    inline void update(float dt){
    	update_positions(dt);
        computeAccels();
        update_velocities(dt);
    }
    inline std::vector<Vec3> getData() {
        std::vector<Vec3> data(N, Vec3());
        for (int i = 0; i < N; i++) {
			data[i] = atoms[i].getPosition();
        }
        return data;
    }
    /*
    std::pair <std::vector<std::vector<Vec3>>,std::vector<std::array<float,2>>> run(int steps,float dt) {
    	std::vector<std::vector<Vec3>> data(steps, std::vector<Vec3>(N, Vec3()));
        std::vector<std::array<float,2>> energies(steps, std::array<float,2>());
        //calculation of v1/2
        computeAccels();
        update_velocities(dt/2);
        //simulation
        const int barWidth = 70;
    	for (int i = 0; i < steps; i++) {
            // print the progress every percent
    	    if (i % (steps/100) == 0) {
                std::cout << "[";
                int pos = barWidth * i / steps;
                for (int j = 0; j < barWidth; ++j) {
                    if (j < pos) std::cout << "=";
                    else if (j == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(i * 100.0 / steps) << " %\r";
                std::cout.flush();
            }
        	update(dt);
            data[i] = getData();
            energies[i][0] = std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
            energies[i][1] = std::accumulate(E_kin.begin(), E_kin.end(), 0.0);
        }
        std::cout << "[" << std::string(barWidth, '=') << "] 100%\n";
        return std::make_pair(data,energies);
    }
     */
    void run(int steps, float dt, char *filename, int resolution) {
        std::ofstream file(filename);
        file << system_size * Sigma * nm << "\n" << T_init << "\n" << N <<  "\n" << steps << "\n" << resolution << "\n";
    	std::vector<Vec3> data(N, Vec3());
        std::array<float,2> energies{0,0};
        //calculation of v1/2
        computeAccels();
        update_velocities(dt/2);
        //simulation
        const int barWidth = 70;
    	for (int i = 0; i < steps; i++) {
            // print the progress every percent
            if (i % (steps/100) == 0) {
                std::cout << "[";
                int pos = barWidth * i / steps;
                for (int j = 0; j < barWidth; ++j) {
                    if (j < pos) std::cout << "=";
                    else if (j == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << int(i * 100.0 / steps) << " %\r";
                std::cout.flush();
            }
        	update(dt);
            data = getData();
            energies[0] = std::accumulate(E_pot.begin(), E_pot.end(), 0.0) * Epsilon * kB;
			energies[1] = std::accumulate(E_kin.begin(), E_kin.end(), 0.0) * Epsilon * kB;
            if (i % resolution == 0) {
                for (int j = 0; j < N; j++) {
                    file << data[j].getX() * Sigma * nm << " " << data[j].getY() * Sigma * nm << " " << data[j].getZ() * Sigma * nm << "\n";
                }
                file << energies[0] << " " << energies[1] << std::endl;
            }
        }
        file.close();
        std::cout << "[" << std::string(barWidth, '=') << "] 100%\n";
    }
    inline void updatePosition(int atom_idx, Vec3 position) {
        atoms[atom_idx].setPosition(position);
    }
};

#endif //VERLET_HPP
