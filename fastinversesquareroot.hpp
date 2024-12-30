#ifndef FASTINVERSESQUAREROOT_HPP
#define FASTINVERSESQUAREROOT_HPP
#include <iostream>
#include <cmath>
#include <iomanip>

union FloatIntUnion {
    float f;
    int i;
};

float Q_rsqrt(float number)
{
    FloatIntUnion u;
    u.f = number;
    u.i = 0x5f3759df - (u.i >> 1);
    u.f = u.f * (1.5f - 0.5f * number * u.f * u.f);
    //u.f = u.f * (1.5f - 0.5f * number * u.f * u.f);
    //u.f = u.f * (1.5f - 0.5f * number * u.f * u.f);
    return u.f;
}

#endif //FASTINVERSESQUAREROOT_HPP
