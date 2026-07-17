#ifndef VERLETIMPL_TPP
#define VERLETIMPL_TPP
template <size_t box_N, size_t N>
System<box_N, N>::System(const double _system_size,
                         const std::vector<Vec3> &_positions,
                         const std::vector<Vec3> &_velocities, double _T_init)
    : system_size(_system_size), T_init(_T_init),
      average_atoms_per_cell(N / (box_N * box_N * box_N)) {
  if (_positions.size() != _velocities.size()) {
    throw std::invalid_argument("Positions and velocities do not match.");
  }
  if (_positions.size() != N) {
    throw std::invalid_argument("Positions and velocities do not match.");
  }
  std::fill(head.begin(), head.end(), EMPTY);
  for (size_t i = 0; i < N; i++) {
    Vec3 position = _positions[i];
    Vec3 velocity = _velocities[i];
    atoms[i] = Atom(position, velocity);
    size_t c = getCellIdx(position);
    next[i] = head[c];
    head[c] = i;
  }
}

template <size_t box_N, size_t N>
[[nodiscard]] size_t System<box_N, N>::getCellIdx(const Vec3 &position) const {
  size_t index = 0;
  index += std::floor(position.x / box_L);
  index += static_cast<size_t>(std::floor(position.y / box_L)) * box_N;
  index += static_cast<size_t>(std::floor(position.z / box_L)) * box_N * box_N;
  return index;
}

template <size_t box_N, size_t N> void System<box_N, N>::computeAccels() {
  std::fill(accels.begin(), accels.end(), Vec3());
  std::fill(E_pot.begin(), E_pot.end(), 0);
  virial = 0;
  std::array<size_t, 26> neighboring_cells{};
  for (size_t cell_index = 0; cell_index < box_N * box_N * box_N;
       ++cell_index) {
    if (head[cell_index] == EMPTY)
      continue;
    neighboring_cells = getNeighboringCells(cell_index);
    // Compute interactions within the same cell
    for (size_t atom_i = head[cell_index]; atom_i != EMPTY;
         atom_i = next[atom_i]) {
      for (size_t atom_j = next[atom_i]; atom_j != EMPTY;
           atom_j = next[atom_j]) {
        const Vec3 ri = atoms[atom_i].getPosition();
        const Vec3 rj = atoms[atom_j].getPosition();
        const Vec3 r = ri - rj;
        const double r2 = r.norm2();
        const Vec3 accel = r * ComputeAccel(r2);
        const double pot = LennardJones(r2);
        accels[atom_i] += accel;
        accels[atom_j] -= accel; // Newton's Third Law
        // Lennard Jones potential
        E_pot[atom_i] += pot / 2;
        E_pot[atom_j] += pot / 2;
        virial += r2 * accel.norm2();
      }
    }
    // Compute interactions with neighboring cells
    for (const size_t neighbor_cell_idx : neighboring_cells) {
      for (size_t atom_i = head[cell_index]; atom_i != EMPTY;
           atom_i = next[atom_i]) {
        for (size_t atom_j = head[neighbor_cell_idx]; atom_j != EMPTY;
             atom_j = next[atom_j]) {
          const Vec3 r =
              PeriodicDifference(atoms[atom_i].getPosition(),
                                 atoms[atom_j].getPosition(), system_size);
          const double r2 = r.norm2();
          const Vec3 accel = r * ComputeAccel(r2);
          const double pot = LennardJones(r2);
          accels[atom_i] += accel;
          E_pot[atom_i] += pot / 2;
          virial += r2 * accel.norm2();
        }
      }
    }
  }
}

template <size_t box_N, size_t N>
double System<box_N, N>::computePotentialEnergy() {
  std::fill(E_pot.begin(), E_pot.end(), 0);
  for (size_t i = 0; i < N; i++) {
    for (size_t j = i + 1; j < N; j++) {
      const Vec3 r = PeriodicDifference(atoms[i].getPosition(),
                                        atoms[j].getPosition(), system_size);
      const double r2 = r.norm2();
      const double pot = LennardJones(r2);
      E_pot[i] += pot / 2;
      virial += r2 * ComputeAccel(r2);
    }
  }
  return std::accumulate(E_pot.begin(), E_pot.end(), 0.0);
}

template <size_t box_N, size_t N>
void System<box_N, N>::update_positions(const double &dt) {
  for (size_t i = 0; i < N; i++) {
    Vec3 position = atoms[i].getPosition();
    const Vec3 velocity = atoms[i].getVelocity();
    // transform the position according to periodic boundary conditions
    position = PeriodicPositionUpdate(position, velocity, dt);
    size_t new_cell_idx = getCellIdx(position);
    atoms[i].setPosition(position);
  }
  std::fill(head.begin(), head.end(), EMPTY);
  for (size_t i = 0; i < N; i++) {
    const size_t cell_idx = getCellIdx(atoms[i].getPosition());
    next[i] = head[cell_idx];
    head[cell_idx] = i;
  }
}

template <size_t box_N, size_t N>
void System<box_N, N>::update_velocities(const double &dt) {
  for (size_t i = 0; i < N; i++) {
    Vec3 velocity = atoms[i].getVelocity();
    const Vec3 accel = accels[i];
    velocity = velocity + accel * dt;
    atoms[i].setVelocity(velocity);
    const double v2 = velocity.norm2();
    E_kin[i] = 0.5 * v2;
  }
}

template <size_t box_N, size_t N>
void System<box_N, N>::update(const double &dt) {
  update_positions(dt);
  computeAccels();
  update_velocities(dt);
}

#endif
