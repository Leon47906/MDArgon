#include "verlet.hpp"

UniformRandomFloat::UniformRandomFloat() : gen(rd()), dis(0, 1) {}
UniformRandomFloat::UniformRandomFloat(const int &seed)
    : gen(seed), dis(0, 1) {}
double UniformRandomFloat::operator()() { return dis(gen); }

NormalRandomFloat::NormalRandomFloat(const double sigma)
    : gen(rd()), dis(0, sigma) {}
NormalRandomFloat::NormalRandomFloat(const size_t &seed, const double sigma)
    : gen(seed), dis(0, sigma) {}
double NormalRandomFloat::operator()() { return dis(gen); }
