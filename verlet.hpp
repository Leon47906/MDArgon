#ifndef VERLET_HPP
#define VERLET_HPP
#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <math.h>
#include <chrono>

// Konstanten

constexpr float kB = 1.38064852e-23; // Boltzmann constant
constexpr float nm = 1e-9; // nanometer
constexpr float ns = 1e-9; // nanosecond
constexpr float fs = 1e-15; // femtosecond
constexpr float Dalton = 1.66053906660e-27; //Dalton in kg
constexpr float Sigma = 0.33916; // Sigma in nm
constexpr float Epsilon = 137.9; // Epsilon in kB*K
constexpr float shift = -0.016316891136; // potential shift
constexpr float Mass = 39.948; // mass of Argon in Dalton
constexpr float one_over_sqrt_pi = 0.5*M_2_SQRTPI;

// Potential

inline float LennardJones(const float& r2) {
	if (r2 > 6.25) return 0;
  	float r6 = r2*r2*r2;
    return 4.0/r6*(1.0/r6 - 1.0)-shift;
}

// Beschleunigung

inline float ComputeAccel(const float& r2) {
    if (r2 > 6.25) return 0;
    float r6 = r2 * r2 * r2;
    float r8 = r6 * r2;
    return 24.0 / r8 * (2.0 / r6 - 1.0);
}

// Zufallszahlengenerator

class UniformRandomFloat{
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis;
    public:
    UniformRandomFloat() : gen(rd()), dis(0, 1) {}
    explicit UniformRandomFloat(const int& seed) : gen(seed), dis(0, 1) {}
    float operator()() {
        return dis(gen);
    }
};

// Dreiervektor

struct Vec3{
	float x, y, z;
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
        return {x + v.x, y + v.y, z + v.z};
    }
    Vec3& operator+=(const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vec3 operator-(const Vec3& v) const {
        return {x - v.x, y - v.y, z - v.z};
    }
    Vec3& operator-=(const Vec3& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    Vec3 operator*(float s) const {
        return {x * s, y * s, z * s};
    }
    friend Vec3 operator*(float s, const Vec3& v) {
        return v * s;
    }
	Vec3 operator/(float s) const {
		if (s == 0) {
			throw std::invalid_argument("Division by zero.");
		}
		return {x / s, y / s, z / s};
	}
	friend Vec3 operator/(float s, const Vec3& v) {
		return v / s;
	}
    [[nodiscard]] float norm2() const { return x*x + y*y + z*z; }
    static Vec3 Zero() { return {0, 0, 0}; }
};

inline float dot(const Vec3& v1, const Vec3& v2) {
    return v1.x() * v2.x() + v1.y() * v2.y() + v1.z() * v2.z();
}

const static std::vector<Vec3> unit_velocities{Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1), Vec3(-1,0,0), Vec3(0,-1,0), Vec3(0,0,-1)};

// Periodische Randbedingungen

inline Vec3 PeriodicDifference(const Vec3& r1,const Vec3& r2, const float& period) {
        const Vec3 r = r1 - r2;
        float x = r.x();
        float y = r.y();
        float z = r.z();
        x -= period * std::round(x / period);
        y -= period * std::round(y / period);
        z -= period * std::round(z / period);
        return Vec3(x, y, z);
    }

// Atom Klasse

class Atom{
	Vec3 position, velocity;
    public:
    Atom() : position(Vec3()), velocity(Vec3()) {}
    Atom(const Vec3& _position, const Vec3& _velocity) : position(_position), velocity(_velocity) {}
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

// System Klasse

class System{
    float system_size, virial;
    int box_N=std::ceil(system_size/2.5), N;
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
            index += std::floor(position.x() / box_L);
            index += std::floor(position.y() / box_L) * box_N;
            index += std::floor(position.z() / box_L) * box_N * box_N;
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
    int getN() const { return N; }
    std::vector<Atom> getAtoms() const { return atoms; }
    int getBoxN() const { return box_N; }
    std::vector<std::vector<int>> getCells() const { return cells; }
    float getSystemSize() const { return system_size; }
    float getPotentialEnergy() const {
        return std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
    }
    std::vector<float> getPotentialEnergies() const { return E_pot; }
    void updatePotentialEnergies(const std::vector<float>& new_potentials) {
        	E_pot = new_potentials;
    }
    // Funktion, welche die Zelle eines Atoms bestimmt
    int getCell(Vec3 position) const {
        int index = 0;
        index += std::floor(position.x() / box_L);
        index += std::floor(position.y() / box_L) * box_N;
        index += std::floor(position.z() / box_L) * box_N * box_N;
        return index;
    }
    Atom getAtom(int atom_index) const {
        return atoms[atom_index];
    }
    // Funktion, welche die Nachbarzellen einer Zelle bestimmt
    std::vector<int> getNeighboringCells(int cell_index) const {
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
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        return neighbors;
    }
    // Funktion, welche die Indizes der Atome in einer Zelle zurückgibt
    std::vector<int> getAtomsInCell(int cell_index) const {
        return cells[cell_index];
    }
    // Funktion, welche die Indizes der Atome in den Nachbarzellen einer Zelle zurückgibt
    std::vector<int> getAtomsInNeighboringCells(int cell_index) const {
        std::vector<int> neighbors = getNeighboringCells(cell_index);
        std::vector<int> atoms_in_neighbors;
        for (int neighbor : neighbors) {
            std::vector<int> atoms_in_cell = getAtomsInCell(neighbor);
            atoms_in_neighbors.insert(atoms_in_neighbors.end(), atoms_in_cell.begin(), atoms_in_cell.end());
        }
        return atoms_in_neighbors;
    }
    // Funktion, welche die Indizes der Atome zurückgibt, die mit einem Atom in Wechselwirkung stehen
    std::vector<int> getAdjacentAtoms(int atom_index) const{
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
    std::vector<int> getAdjacentAtoms(const Vec3& position) const {
        const int cell_index = getCell(position);
        std::vector<int> adjacent_atoms;
        for (int atom : getAtomsInCell(cell_index)) {
            adjacent_atoms.push_back(atom);
        }
        std::vector<int> neighbors = getAtomsInNeighboringCells(cell_index);
        adjacent_atoms.insert(adjacent_atoms.end(), neighbors.begin(), neighbors.end());
        return adjacent_atoms;
    }
    // Debugging Funktionen
	void show_neighboring_cells(int cell_index) const{
		std::vector<int> neighbors = getNeighboringCells(cell_index);
        std::cout << "Neighboring cells of cell " << cell_index << " are: ";
        for (int neighbor : neighbors) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
	}
    void show_adjacent_atoms(int atom_index) const {
		std::vector<int> adjacent_atoms = getAdjacentAtoms(atom_index);
        std::cout << "Adjacent atoms of atom " << atom_index << " are: ";
        for (int atom : adjacent_atoms) {
            std::cout << atom << " ";
        }
        std::cout << std::endl;
    }
    void display() const {
        for (int i = 0; i < N; i++) {
            std::cout << "Atom " << i << " position: " << atoms[i].getPosition().x() << " "
                      << atoms[i].getPosition().y() << " " << atoms[i].getPosition().z() << std::endl;
            std::cout << "Atom " << i << " velocity: " << atoms[i].getVelocity().x() << " "
                      << atoms[i].getVelocity().y() << " " << atoms[i].getVelocity().z() << std::endl;
        }
    }
    // Funktion, welche die Position eines Atoms transformiert, um periodische Randbedingungen zu berücksichtigen
    Vec3 PeriodicPositionUpdate(const Vec3& position, const Vec3& velocity, const float& dt) const {
        Vec3 new_position = position + velocity * dt;
        float x = new_position.x();
        float y = new_position.y();
        float z = new_position.z();
        x = std::fmod(x + system_size, system_size);
        y = std::fmod(y + system_size, system_size);
        z = std::fmod(z + system_size, system_size);
        return Vec3(x, y, z);
    }
    // Funktion, welche die Beschleunigungen der Atome aufgrund der Lennard-Jones-Kräfte berechnet
    void computeAccels() {
        std::fill(accels.begin(), accels.end(), Vec3());
        std::fill(E_pot.begin(), E_pot.end(), 0);
        virial = 0;
        for (int cell = 0; cell < box_N * box_N * box_N; ++cell) {
            const std::vector<int>& cell_atoms = cells[cell];
            const std::vector<int>& neighboring_cells = getNeighboringCells(cell);
            if (cell_atoms.empty()) continue;
            // Compute interactions within the same cell
            const int N = cell_atoms.size();
            for (int i = 0; i < N; i++) {
                for (int j = i + 1; j < N; j++) {
                    const int atom_i = cell_atoms[i];
                    const int atom_j = cell_atoms[j];
                    Vec3 ri = atoms[atom_i].getPosition();
                    Vec3 rj = atoms[atom_j].getPosition();
                    Vec3 r = ri - rj;
                    float r2 = r.norm2();
                    Vec3 accel = r * ComputeAccel(r2);
                    const float pot = LennardJones(r2);
                    accels[atom_i] += accel;
                    accels[atom_j] -= accel; // Newton's Third Law
                    // Lennard Jones potential
                    E_pot[atom_i] += pot/2;
                    E_pot[atom_j] += pot/2;
                    virial += r2*accel.norm2();
                }
            }
            // Compute interactions with neighboring cells
            for (const int neighbor_cell : neighboring_cells) {
                const std::vector<int>& neighbor_atoms = cells[neighbor_cell];
                for (const int atom_i : cell_atoms) {
                    for (const int atom_j : neighbor_atoms) {
                        Vec3 r = PeriodicDifference(atoms[atom_i].getPosition(), atoms[atom_j].getPosition(),system_size);
                        float r2 = r.norm2();
                        Vec3 accel = r * ComputeAccel(r2);
                        const float pot = LennardJones(r2);
                        accels[atom_i] += accel;
                        E_pot[atom_i] += pot/2;
                        virial += r2*accel.norm2();
                    }
                }
            }
        }
    }
    float computePotentialEnergy() {
        std::fill(E_pot.begin(), E_pot.end(), 0);
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                Vec3 r = PeriodicDifference(atoms[i].getPosition(), atoms[j].getPosition(), system_size);
                float r2 = r.norm2();
                const float pot = LennardJones(r2);
                E_pot[i] += pot/2;
                virial += r2*ComputeAccel(r2);
            }
        }
        return std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
    }
    // Funktion, welche die Positionen der Atome aktualisiert
	void update_positions(const float& dt){
		for (int i = 0; i < N; i++) {
        	Vec3 position = atoms[i].getPosition();
            Vec3 velocity = atoms[i].getVelocity();
            //transform the position according to periodic boundary conditions
            position = PeriodicPositionUpdate(position,velocity,dt);
            atoms[i].setPosition(position);
    	}
    }
    // Funktion, welche die Geschwindigkeiten der Atome aktualisiert
    void update_velocities(const float& dt){
    	for (int i = 0; i < N; i++) {
        	Vec3 velocity = atoms[i].getVelocity();
            Vec3 accel = accels[i];
            velocity = velocity + accel * dt;
            atoms[i].setVelocity(velocity);
            float v2 = velocity.norm2();
            E_kin[i] = 0.5 * v2;
        }
    }
    // Funktion, welche einen Zeitschritt des Verlet-Algorithmus durchführt
    void update(const float dt){
    	update_positions(dt);
        computeAccels();
        update_velocities(dt);
    }
    // Funktion, welche die Daten der Atome zurückgibt
    std::vector<Vec3> getData() const {
        std::vector<Vec3> data(N, Vec3());
        for (int i = 0; i < N; i++) {
			data[i] = atoms[i].getPosition();
        }
        return data;
    }
    // Funktion, welche die Simulation durchführt und die Daten in eine Datei schreibt
    void run(const int& steps, const float& dt, const char *filename,
             const int& resolution) {
        std::ofstream file(filename);
        file << system_size * Sigma * nm << "\n" << T_init * Epsilon << "\n" << N <<  "\n" << steps << "\n" << resolution << "\n" << dt << "\n";
    	std::vector<Vec3> data(N, Vec3());
        std::array<float,2> energies{0,0};
        //calculation of v1/2
        computeAccels();
        update_velocities(dt/2);
        // simulation
        constexpr int barWidth = 70;
    	for (int i = 0; i < steps; i++) {
            // print the progress every percent
            if (i % (steps/100) == 0) {
                std::cout << "[";
                const int pos = barWidth * i / steps;
                for (int j = 0; j < barWidth; ++j) {
                    if (j < pos) std::cout << "=";
                    else if (j == pos) std::cout << ">";
                    else std::cout << " ";
                }
                std::cout << "] " << static_cast<int>(i * 100.0 / steps) << " %\r";
                std::cout.flush();
            }
        	update(dt);
            data = getData();
            energies[0] = std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
			energies[1] = std::accumulate(E_kin.begin(), E_kin.end(), 0.0);
	        // give out an error if any of the energies is nan
			if (energies[0] != energies[0] || energies[1] != energies[1]) {
                std::cerr << "Error: Energy is NaN" << std::endl;
				break;
            }
            if (i % resolution == 0) {
                for (int j = 0; j < N; j++) {
                    file << data[j].x() * Sigma * nm << " " << data[j].y() * Sigma * nm << " " << data[j].z() * Sigma * nm << "\n";
                }
                file << energies[0] << " " << energies[1] << "\n";
                file << virial << std::endl;
            }
        }
        file.close();
        std::cout << "[" << std::string(barWidth, '=') << "] 100%\n";
    }
    // Funktion, welche die Position eines Atoms aktualisiert
    void updatePosition(int atom_idx, Vec3 position) {
        atoms[atom_idx].setPosition(position);
    }
};

#endif //VERLET_HPP
