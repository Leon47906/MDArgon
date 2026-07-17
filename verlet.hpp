#ifndef VERLET_HPP
#define VERLET_HPP
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// Konstanten

constexpr double kB = 1.38064852e-23;        // Boltzmann constant
constexpr double nm = 1e-9;                  // nanometer
constexpr double ns = 1e-9;                  // nanosecond
constexpr double fs = 1e-15;                 // femtosecond
constexpr double Dalton = 1.66053906660e-27; // Dalton in kg
constexpr double Sigma = 0.33916;            // Sigma in nm
constexpr double Epsilon = 137.9;            // Epsilon in kB*K
constexpr double shift = -0.016316891136;    // potential shift
constexpr double Mass = 39.948;              // mass of Argon in Dalton
constexpr double one_over_sqrt_pi = 0.5 * M_2_SQRTPI;

constexpr size_t EMPTY = std::numeric_limits<size_t>::max();

// Potential

inline double LennardJones(const double &r2) {
  if (r2 > 6.25)
    return 0;
  const double r6 = r2 * r2 * r2;
  return 4.0 / r6 * (1.0 / r6 - 1.0) - shift;
}

// Beschleunigung

inline double ComputeAccel(const double &r2) {
  if (r2 > 6.25)
    return 0;
  const double r6 = r2 * r2 * r2;
  const double r8 = r6 * r2;
  return 24.0 / r8 * (2.0 / r6 - 1.0);
}

// Zufallszahlengenerator

class UniformRandomFloat {
  std::random_device rd;
  std::mt19937 gen;
  std::uniform_real_distribution<double> dis;

public:
  UniformRandomFloat();
  explicit UniformRandomFloat(const int &seed);
  double operator()();
};

class NormalRandomFloat {
  std::random_device rd;
  std::mt19937 gen;
  std::normal_distribution<double> dis;

public:
  explicit NormalRandomFloat(const double sigma);
  NormalRandomFloat(const size_t &seed, const double sigma);
  double operator()();
};

// Dreiervektor

struct Vec3 {
  double x, y, z;
  Vec3() : x(0), y(0), z(0) {}
  Vec3(const double _x, const double _y, const double _z)
      : x(_x), y(_y), z(_z) {}
  Vec3(const Vec3 &v) = default;
  Vec3 &operator=(const Vec3 &v) = default;
  Vec3 operator+(const Vec3 &v) const { return {x + v.x, y + v.y, z + v.z}; }
  Vec3 &operator+=(const Vec3 &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
  }
  Vec3 operator-(const Vec3 &v) const { return {x - v.x, y - v.y, z - v.z}; }
  Vec3 &operator-=(const Vec3 &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
  }
  Vec3 operator*(const double s) const { return {x * s, y * s, z * s}; }
  friend Vec3 operator*(const double s, const Vec3 &v) { return v * s; }
  Vec3 operator/(const double s) const {
    if (s == 0) {
      throw std::invalid_argument("Division by zero.");
    }
    return {x / s, y / s, z / s};
  }
  friend Vec3 operator/(double s, const Vec3 &v) { return v / s; }
  [[nodiscard]] double norm2() const { return x * x + y * y + z * z; }
  static Vec3 Zero() { return {0, 0, 0}; }
};

inline double dot(const Vec3 &v1, const Vec3 &v2) {
  return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// Periodische Randbedingungen

inline Vec3 PeriodicDifference(const Vec3 &r1, const Vec3 &r2,
                               const double &period) {
  const Vec3 r = r1 - r2;
  double x = r.x;
  double y = r.y;
  double z = r.z;
  x -= period * std::round(x / period);
  y -= period * std::round(y / period);
  z -= period * std::round(z / period);
  return {x, y, z};
}

// Atom Klasse

class Atom {
  Vec3 position, velocity;

public:
  Atom() : position(Vec3()), velocity(Vec3()) {}
  Atom(const Vec3 &_position, const Vec3 &_velocity)
      : position(_position), velocity(_velocity) {}
  Atom(const Atom &other) = default;
  Atom &operator=(const Atom &other) {
    if (this != &other) {
      position = other.position;
      velocity = other.velocity;
    }
    return *this;
  }
  // Copy assignment operator

  [[nodiscard]] Vec3 getPosition() const { return position; }
  [[nodiscard]] Vec3 getVelocity() const { return velocity; }

  void setPosition(const Vec3 &position) { this->position = position; }
  void setVelocity(const Vec3 &velocity) { this->velocity = velocity; }
};

// System Klasse

template <size_t box_N, size_t N> class System {
  double virial{};
  const double system_size;
  // size_t box_N=std::ceil(system_size/2.5), N;
  const double box_L = system_size / static_cast<double>(box_N);
  const size_t average_atoms_per_cell;
  // std::vector<Cell<N>> cells;
  std::array<size_t, box_N * box_N * box_N> head;
  std::array<size_t, N> next;
  std::array<Atom, N> atoms;
  std::array<Vec3, N> accels;
  std::array<double, N> E_pot, E_kin;
  double T_init;

public:
  System(const double _system_size, const std::vector<Vec3> &_positions,
         const std::vector<Vec3> &_velocities, double _T_init);
  [[nodiscard]] static size_t getN() { return N; }
  [[nodiscard]] std::vector<Atom> getAtoms() const { return atoms; }
  [[nodiscard]] static size_t getBoxN() { return box_N; }
  [[nodiscard]] auto getHead() const { return head; }
  [[nodiscard]] auto getNext() const { return next; }
  [[nodiscard]] double getSystemSize() const { return system_size; }
  [[nodiscard]] double getPotentialEnergy() const {
    return std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
  }
  [[nodiscard]] auto getPotentialEnergies() const { return E_pot; }
  void updatePotentialEnergies(const std::array<double, N> &new_potentials) {
    E_pot = new_potentials;
  }
  // Funktion, welche die Zelle eines Atoms bestimmt
  [[nodiscard]] size_t getCellIdx(const Vec3 &position) const;
  [[nodiscard]] Atom getAtom(const size_t atom_index) const {
    return atoms[atom_index];
  }
  // Funktion, welche die Nachbarzellen einer Zelle bestimmt
  [[nodiscard]] static std::array<size_t, 26>
  getNeighboringCells(const size_t cell_index) {
    std::array<size_t, 26> neighbors{};
    size_t index = 0;
    const size_t x = cell_index % box_N;
    const size_t y = (cell_index / box_N) % box_N;
    const size_t z = (cell_index / box_N / box_N) % box_N;
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        for (int k = -1; k <= 1; k++) {
          const size_t x_neighbor = (x + i + box_N) % box_N;
          const size_t y_neighbor = (y + j + box_N) % box_N;
          const size_t z_neighbor = (z + k + box_N) % box_N;
          if (const size_t neighbor_index =
                  x_neighbor + y_neighbor * box_N + z_neighbor * box_N * box_N;
              cell_index != neighbor_index) {
            neighbors[index] = neighbor_index;
            index++;
          }
        }
      }
    }
    return neighbors;
  }
  // Funktion, welche die Indizes der Atome in den Nachbarzellen einer Zelle
  // zurückgibt
  [[nodiscard]] std::vector<size_t>
  getAtomsInNeighboringCells(const size_t cell_index) const {
    const auto neighbors = getNeighboringCells(cell_index);
    std::vector<size_t> atoms_in_neighbors;
    atoms_in_neighbors.reserve(26 * average_atoms_per_cell);
    for (const size_t neighbor : neighbors) {
      for (size_t i = head[neighbor]; i != EMPTY; i = next[i]) {
        atoms_in_neighbors.push_back(i);
      }
    }
    return atoms_in_neighbors;
  }
  // Funktion, welche die Indizes der Atome zurückgibt, die mit einem Atom in
  // Wechselwirkung stehen
  [[nodiscard]] std::vector<size_t>
  getAdjacentAtoms(const size_t atom_index) const {
    const size_t cell_index = getCellIdx(atoms[atom_index].getPosition());
    std::vector<size_t> adjacent_atoms;
    adjacent_atoms.reserve(27 * average_atoms_per_cell);
    for (size_t i = head[cell_index]; i != EMPTY; i = next[i]) {
      adjacent_atoms.push_back(i);
    }
    if (auto it =
            std::find(adjacent_atoms.begin(), adjacent_atoms.end(), atom_index);
        it != adjacent_atoms.end()) {
      adjacent_atoms.erase(it);
    }
    const std::vector<size_t> neighbors =
        getAtomsInNeighboringCells(cell_index);
    adjacent_atoms.insert(adjacent_atoms.end(), neighbors.begin(),
                          neighbors.end());
    return adjacent_atoms;
  }
  [[nodiscard]] std::vector<size_t>
  getAdjacentAtoms(const Vec3 &position) const {
    const size_t cell_index = getCellIdx(position);
    std::vector<size_t> adjacent_atoms;
    adjacent_atoms.reserve(27 * average_atoms_per_cell);
    for (size_t i = head[cell_index]; i != EMPTY; i = next[i]) {
      adjacent_atoms.push_back(i);
    }
    const std::vector<size_t> neighbors =
        getAtomsInNeighboringCells(cell_index);
    adjacent_atoms.insert(adjacent_atoms.end(), neighbors.begin(),
                          neighbors.end());
    return adjacent_atoms;
  }
  // Debugging Funktionen
  void show_neighboring_cells(size_t cell_index) const {
    std::array<size_t, N> neighbors = getNeighboringCells(cell_index);
    std::cout << "Neighboring cells of cell " << cell_index << " are: ";
    for (const size_t neighbor : neighbors) {
      std::cout << neighbor << " ";
    }
    std::cout << std::endl;
  }
  void show_adjacent_atoms(const size_t atom_index) const {
    const std::vector<size_t> adjacent_atoms = getAdjacentAtoms(atom_index);
    std::cout << "Adjacent atoms of atom " << atom_index << " are: ";
    for (const size_t atom : adjacent_atoms) {
      std::cout << atom << " ";
    }
    std::cout << std::endl;
  }
  void display() const {
    for (size_t i = 0; i < N; i++) {
      std::cout << "Atom " << i << " position: " << atoms[i].getPosition().x
                << " " << atoms[i].getPosition().y << " "
                << atoms[i].getPosition().z << std::endl;
      std::cout << "Atom " << i << " velocity: " << atoms[i].getVelocity().x
                << " " << atoms[i].getVelocity().y << " "
                << atoms[i].getVelocity().z << std::endl;
    }
  }
  // Funktion, welche die Position eines Atoms transformiert, um periodische
  // Randbedingungen zu berücksichtigen
  [[nodiscard]] Vec3 PeriodicPositionUpdate(const Vec3 &position,
                                            const Vec3 &velocity,
                                            const double &dt) const {
    const Vec3 new_position = position + velocity * dt;
    double x = new_position.x;
    double y = new_position.y;
    double z = new_position.z;
    x = std::fmod(x + system_size, system_size);
    y = std::fmod(y + system_size, system_size);
    z = std::fmod(z + system_size, system_size);
    return {x, y, z};
  }
  // Funktion, welche die Beschleunigungen der Atome aufgrund der
  // Lennard-Jones-Kräfte berechnet
  void computeAccels();
  double computePotentialEnergy();
  // Funktion, welche die Positionen der Atome aktualisiert
  void update_positions(const double &dt);
  // Funktion, welche die Geschwindigkeiten der Atome aktualisiert
  void update_velocities(const double &dt);
  // Funktion, welche einen Zeitschritt des Verlet-Algorithmus durchführt
  void update(const double &dt);
  // Funktion, welche die Daten der Atome zurückgibt
  [[nodiscard]] std::array<Vec3, N> getData() const {
    std::array<Vec3, N> data;
    for (size_t i = 0; i < N; i++) {
      data[i] = atoms[i].getPosition();
    }
    return data;
  }
  // Funktion, welche die Simulation durchführt und die Daten in eine Datei
  // schreibt
  void run(const size_t &steps, const double &dt, const char *filename,
           const size_t &resolution) {
    std::ofstream file(filename);
    file << system_size * Sigma * nm << "\n"
         << T_init * Epsilon << "\n"
         << N << "\n"
         << steps << "\n"
         << resolution << "\n"
         << dt << "\n";
    std::array<Vec3, N> data;
    std::array<double, 2> energies{0, 0};
    // calculation of v1/2
    computeAccels();
    update_velocities(dt / 2);
    // simulation
    constexpr size_t barWidth = 70;
    for (size_t i = 0; i < steps; i++) {
      // print the progress every percent
      if (i % (steps / 100) == 0) {
        std::cout << "[";
        const size_t pos = barWidth * i / steps;
        for (size_t j = 0; j < barWidth; ++j) {
          if (j < pos)
            std::cout << "=";
          else if (j == pos)
            std::cout << ">";
          else
            std::cout << " ";
        }
        std::cout << "] " << std::setprecision(2)
                  << static_cast<double>(i) * 100.0 / static_cast<double>(steps)
                  << " %\r";
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
        for (size_t j = 0; j < N; j++) {
          file << data[j].x * Sigma * nm << " " << data[j].y * Sigma * nm << " "
               << data[j].z * Sigma * nm << "\n";
        }
        file << energies[0] << " " << energies[1] << "\n";
        file << virial << std::endl;
      }
    }
    file.close();
    std::cout << "[" << std::string(barWidth, '=') << "] 100%\n";
  }
  // Funktion, welche die Position eines Atoms aktualisiert
  void updatePosition(size_t atom_idx, Vec3 position) {
    atoms[atom_idx].setPosition(position);
  }
};

#include "verlet_impl.hpp"
#endif // VERLET_HPP
