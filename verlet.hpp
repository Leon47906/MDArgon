#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <omp.h>
#include "fastinversesquareroot.hpp"

#ifndef VERLET_HPP
#define VERLET_HPP

static constexpr double kB = 1.38064852e-23;
static constexpr double nm = 1e-9;//nanometer
static constexpr double ns = 1e-9;//nanosecond
static constexpr double fs = 1e-15;//femtosecond
static constexpr double Sigma = 0.33916*nm;
static constexpr double Sigma2 = Sigma*Sigma;
static constexpr double Sigma6 = Sigma2*Sigma2*Sigma2;
static constexpr double Sigma12 = Sigma6*Sigma6;
static constexpr double cutoff = 2.5*Sigma;
static constexpr double Epsilon = 137.9*kB;
static constexpr double shift = -4*Epsilon*0.004079222784;
static constexpr double Mass = 6.6335209e-26;
static constexpr double ForceConstant = 24*Epsilon/(Sigma*Mass);
static constexpr double one_over_sqrt_pi = 0.5*M_2_SQRTPI;
static constexpr double Avogadro = 6.02214076e23;
static constexpr double V_min = 10*kB;


double LennardJones(double r2) {
  	double r6 = r2*r2*r2;
    return 4*Epsilon/r6*(Sigma12/r6 - Sigma6)-shift;
}

double vanDerWaalsPressure(double T, double V, double N, double a, double b) {
    // Umrechnungen für die Van-der-Waals-Parameter
    double a_prime = a / (Avogadro * Avogadro);   // Wechselwirkungsparameter pro Teilchenpaar
    double b_prime = b / Avogadro;           // Eigenvolumen pro Teilchen
    // Berechnung des Drucks
    double P = (N * kB * T / (V - N * b_prime)) - (a_prime * N * N / (V * V));
    return P;
}

class UniformRandomDouble{
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    public:
    UniformRandomDouble() : gen(rd()), dis(0, 1) {}
    double operator()() {
        return dis(gen);
    }
};

class Vec3{
	double x, y, z;
    public:
	Vec3() : x(0), y(0), z(0) {}
	Vec3(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
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
    Vec3 operator*(double s) const {
        return Vec3(x * s, y * s, z * s);
    }
    friend Vec3 operator*(double s, const Vec3& v) {
        return v * s;
    }
    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }
    double norm2() const { return x*x + y*y + z*z; }
    Vec3 Zero() const { return Vec3(0, 0, 0); }
};

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

double ComputeAccel(double r2) {
    if (r2 > cutoff*cutoff) return 0;
    else {
    	double r6 = r2*r2*r2;
        double r_inv = 1/std::sqrt(r2);
        return ForceConstant *  Sigma * r_inv / r6 * (2 * Sigma12/r6 - Sigma6);
    }
}

auto pot = [](double u, double system_size) { return std::cos(u/system_size*2*M_PI)-1; };
auto dpot = [](double u, double system_size) { return -1*std::sin(u/system_size*2*M_PI)*2*M_PI/system_size; };
double confiningPotential(const Vec3& position, double L) {
    return 0.25*V_min*(pot(position.getX(),L)+pot(position.getY(),L)+pot(position.getZ(),L));
}

Vec3 confiningAccel(const Vec3& position, double system_size) {
    double x = position.getX();
    double y = position.getY();
    double z = position.getZ();
    double ax = dpot(x,system_size)*pot(y,system_size)*pot(z,system_size);
    double ay = pot(x,system_size)*dpot(y,system_size)*pot(z,system_size);
    double az = pot(x,system_size)*pot(y,system_size)*dpot(z,system_size);
    return -0.25*V_min*2*M_PI/system_size/Mass*Vec3(ax,ay,az);
}


class System{
    double system_size, N;
    int box_N=std::ceil(system_size/cutoff);
    double box_L=system_size/box_N;
    std::vector<std::vector<int>> cells;
    std::vector<Atom> atoms;
    std::vector<Vec3> accels;
    std::vector<double> E_pot, E_kin;
    public:
    System(double _system_size, std::vector<Vec3> _positions, std::vector<Vec3> _velocities) : system_size(_system_size), N(_positions.size()) {
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
    /*
    System(double _system_size, double _N, double _T_init) : system_size(_system_size*nm), N(_N), T_init(_T_init) {
        accels.resize(N*N*N, Vec3());
        E_pot.resize(N*N*N,0);
        E_kin.resize(N*N*N,0);
        cells.resize(box_N*box_N*box_N);
        UniformRandomdouble random;
        // Initialize the atoms in a simple cubic lattice and random direction velocities
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    Vec3 position(system_size * (0.125 + 0.75 * i / (N-1)),
                                  system_size * (0.125 + 0.75 * j / (N-1)),
                                  system_size * (0.125 + 0.75 * k / (N-1)));
                    Vec3 velocity = unit_velocities[std::floor(6*random())]*std::sqrt(8*kB*T_init/Mass)*one_over_sqrt_pi;
                    atoms.push_back(Atom(position, velocity));
                    // Add the atom to the cell by index
                    int index = 0, atom_index = 0;
                    index += std::floor(position.getX() / box_L);
                    index += std::floor(position.getY() / box_L) * box_N;
                    index += std::floor(position.getZ() / box_L) * box_N * box_N;
                    atom_index = i + j * N + k * N * N;
                    cells[index].push_back(atom_index);
                }
            }
        }
    }
     */
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
    double getSystemSize() const { return system_size; }
    double getPotentialEnergy() const {
        return std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
    }
    // I want to implement a function, that give me the indices of the neighboring cells
    int getCell(Vec3 position) const {
        //Vec3 position = atoms[atom_index].getPosition();
        int index = 0;
        index += std::floor(position.getX() / box_L);
        index += std::floor(position.getY() / box_L) * box_N;
        index += std::floor(position.getZ() / box_L) * box_N * box_N;
        return index;
    }
    Atom getAtom(int atom_index) const {
        return atoms[atom_index];
    }
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
        //return the sorted list of neighbors
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        return neighbors;
    }
    std::vector<int> getAtomsInCell(int cell_index) const {
        return cells[cell_index];
    }
    std::vector<int> getAtomsInNeighboringCells(int cell_index) const {
        std::vector<int> neighbors = getNeighboringCells(cell_index);
        std::vector<int> atoms_in_neighbors;
        for (int neighbor : neighbors) {
            std::vector<int> atoms_in_cell = getAtomsInCell(neighbor);
            atoms_in_neighbors.insert(atoms_in_neighbors.end(), atoms_in_cell.begin(), atoms_in_cell.end());
        }
        return atoms_in_neighbors;
    }
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
    Vec3 PeriodicDifferece(const Vec3& r1,const Vec3& r2, const double& period) const {
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
    Vec3 PeriodicPositionUpdate(const Vec3& position, const Vec3& velocity, const double& dt) const {
        Vec3 new_position = position + velocity * dt;
        double x = new_position.getX();
        double y = new_position.getY();
        double z = new_position.getZ();
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
                    double r2 = r.norm2();
                    Vec3 accel = r*ComputeAccel(r2);
                    double pot = LennardJones(r2);
                    // Lennard Jones potential
                    E_pot[atom_i] += pot;
                    E_pot[atom_j] += pot;
                    accels[atom_i] += accel;
                    accels[atom_j] -= accel; // Newton's Third Law
                    // Confining potential
                    //E_pot[atom_i] += confiningPotential(ri, system_size);
                    //E_pot[atom_j] += confiningPotential(rj, system_size);
                    //accels[atom_i] += confiningAccel(ri, system_size);
                    //accels[atom_j] += confiningAccel(rj, system_size);

                    }
                }
            }
            // Compute interactions with neighboring cells
            for (int neighbor_cell : neighboring_cells) {
                const std::vector<int>& neighbor_atoms = cells[neighbor_cell];
                for (int atom_i : cell_atoms) {
                    for (int atom_j : neighbor_atoms) {
                        Vec3 r = PeriodicDifferece(atoms[atom_i].getPosition(), atoms[atom_j].getPosition(),system_size);
                        double r2 = r.norm2();
                        Vec3 accel = r*ComputeAccel(r2);
                        double pot = LennardJones(r2);
                        accels[atom_i] += accel;
                        E_pot[atom_i] += pot;
                    }
                }
            }
        }
    }

    //Parallel version of the computeAccels function
    /*
	void computeAccels() {
    	// Reset accelerations and potential energies
    	for (int i = 0; i < N; i++) {
        	accels[i] = Vec3();
        	E_pot[i] = 0;
    	}
		int num_threads;
		#pragma omp parallel
        {
            num_threads = omp_get_num_threads();
        }

        // Thread-local acceleration buffers
        std::vector<std::vector<Vec3>> thread_accels(num_threads, std::vector<Vec3>(atoms.size(), Vec3(0, 0, 0)));
        // Thread-local potential energy buffers
        std::vector<std::vector<double>> thread_E_pot(num_threads, std::vector<double>(atoms.size(), 0));
        // Compute accelerations in parallel
		#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
    		// Parallelize over cells
    		#pragma omp for schedule(dynamic)
    		for (int cell = 0; cell < box_N * box_N * box_N; ++cell) {
        		const std::vector<int>& cell_atoms = cells[cell];
        		const std::vector<int>& neighboring_cells = getNeighboringCells(cell);
        		if (cell_atoms.empty()) continue;
        		// Compute interactions within the same cell
        		for (int i = 0; i < cell_atoms.size(); ++i) {
            		int atom_i = cell_atoms[i];
            		for (int j = i + 1; j < cell_atoms.size(); ++j) { // Avoid double-counting
            	    	int atom_j = cell_atoms[j];
            	    	Vec3 ri = atoms[atom_i].getPosition();
            	    	Vec3 rj = atoms[atom_j].getPosition();
            	    	Vec3 r = ri - rj;
            	    	double r2 = r.getX() * r.getX() + r.getY() * r.getY() + r.getZ() * r.getZ();
                		Vec3 accel = r * ComputeAccel(r2);
                		double pot = LennardJones(r2);
        	        	thread_E_pot[thread_id][atom_i] += pot;
        	        	thread_E_pot[thread_id][atom_j] += pot;
        	        	thread_accels[thread_id][atom_i] += accel;
        	        	thread_accels[thread_id][atom_j] -= accel; // Newton's Third Law
        	    	}
        		}
        		// Compute interactions with neighboring cells
        		for (int neighbor_cell : neighboring_cells) {
            		const std::vector<int>& neighbor_atoms = cells[neighbor_cell];
            		for (int atom_i : cell_atoms) {
            	    	for (int atom_j : neighbor_atoms) {
            	        	Vec3 r = PeriodicDifferece(atoms[atom_i].getPosition(), atoms[atom_j].getPosition(), system_size);
            	        	double r2 = r.getX() * r.getX() + r.getY() * r.getY() + r.getZ() * r.getZ();
            	        	Vec3 accel = r * ComputeAccel(r2);
            	        	double pot = LennardJones(r2);
            	        	thread_E_pot[thread_id][atom_i] += pot;
            	        	thread_accels[thread_id][atom_i] += accel;
            	    	}
            		}
        		}
        		for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
            		for (int i = 0; i < atoms.size(); ++i) {
                		E_pot[i] += thread_E_pot[thread_id][i];
                		accels[i] += thread_accels[thread_id][i];
            		}
        		}
        	}
    	}
    }
    */
    double computePotentialEnergy() {
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
                    double r2 = r.norm2();
                    double pot = LennardJones(r2);
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
                        double r2 = r.norm2();
                        double pot = LennardJones(r2);
                        E_pot[atom_i] += pot;
                    }
                }
            }
        }
        return std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
    }
    double computePotentialSansAtomidx(const int &atom_idx) const{
        std::vector<double> temp_E_pot = E_pot;
        int cell = getCell(atoms[atom_idx].getPosition());
        const std::vector<int>& cell_atoms = cells[cell];
        const std::vector<int>& neighboring_cells = getNeighboringCells(cell);
        for (int atom_i : cell_atoms) {
            if (atom_i != atom_idx) {
                Vec3 ri = atoms[atom_i].getPosition();
                Vec3 r = atoms[atom_idx].getPosition() - ri;
                double r2 = r.norm2();
                double pot = LennardJones(r2);
                temp_E_pot[atom_i] -= pot;
            }
        }
        for (int neighbor_cell : neighboring_cells) {
            const std::vector<int>& neighbor_atoms = cells[neighbor_cell];
            for (int atom_i : cell_atoms) {
                for (int atom_j : neighbor_atoms) {
                    if (atom_i != atom_idx) {
                        Vec3 r = PeriodicDifferece(atoms[atom_i].getPosition(), atoms[atom_j].getPosition(),system_size);
                        double r2 = r.norm2();
                        double pot = LennardJones(r2);
                        temp_E_pot[atom_i] -= pot;
                    }
                }
            }
        }
        return std::accumulate(temp_E_pot.begin(), temp_E_pot.end(), 0.0);
    }
	void update_positions(double dt){
		for (int i = 0; i < N; i++) {
        	Vec3 position = atoms[i].getPosition();
            Vec3 velocity = atoms[i].getVelocity();
            //transform the position according to periodic boundary conditions
            position = PeriodicPositionUpdate(position,velocity,dt);
            atoms[i].setPosition(position);
    	}
    }
    void update_velocities(double dt){
    	for (int i = 0; i < N; i++) {
        	Vec3 velocity = atoms[i].getVelocity();
            Vec3 accel = accels[i];
            velocity = velocity + accel * dt;
            atoms[i].setVelocity(velocity);
            double v2 = velocity.norm2();
            E_kin[i] = 0.5 * Mass * v2;
        }
    }
    void update(double dt){
    	update_positions(dt);
        computeAccels();
        update_velocities(dt);
    }
    std::vector<Vec3> getData() {
        std::vector<Vec3> data(N, Vec3());
        for (int i = 0; i < N; i++) {
			data[i] = atoms[i].getPosition();
        }
        return data;
    }
    std::pair <std::vector<std::vector<Vec3>>,std::vector<std::array<double,2>>> run(int steps,double dt){
    	std::vector<std::vector<Vec3>> data(steps, std::vector<Vec3>(N, Vec3()));
        std::vector<std::array<double,2>> energies(steps, std::array<double,2>());
        //calculation of v1/2
        computeAccels();
        update_velocities(dt/2);
        //simulation
    	for (int i = 0; i < steps; i++) {
            // print the progress every percent
            if (i % (steps / 100) == 0) {
                std::cout << "Progress: " << i / (steps / 100) << "%" << std::endl;
                /*
                Vec3 total_velocity = Vec3();
                for (int j = 0; j < N; j++) {
                    total_velocity += atoms[j].getVelocity();
                }
                std::cout << "Total velocity: " << total_velocity.getX() << " " << total_velocity.getY() << " " << total_velocity.getZ() << std::endl;
                */
            }
        	update(dt);
            data[i] = getData();
            energies[i][0] = std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
            energies[i][1] = std::accumulate(E_kin.begin(), E_kin.end(), 0.0);
        }
        std::cout << "Progress: 100%" << std::endl;
        return std::make_pair(data,energies);
    }
    void updatePosition(int atom_idx, Vec3 position) {
        atoms[atom_idx].setPosition(position);
    }
};

#endif //VERLET_HPP
