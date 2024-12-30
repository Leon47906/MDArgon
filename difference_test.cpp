//
// Created by leonard on 27.12.24.
//
#include "verlet.hpp"

Vec3 AltPeriodicDifferece(const Vec3& r1, const Vec3& r2, const double& period) {
    Vec3 r = r1 - r2;

    double x = r.getX();
    double y = r.getY();
    double z = r.getZ();

    // Apply the minimum image convention
    while (x > period * 0.5) x -= period;
    while (x < -period * 0.5) x += period;

    while (y > period * 0.5) y -= period;
    while (y < -period * 0.5) y += period;

    while (z > period * 0.5) z -= period;
    while (z < -period * 0.5) z += period;

    return Vec3(x, y, z);
}

Vec3 PeriodicPositionUpdate(const Vec3& position, const Vec3& velocity, const double& dt, const double& period) {
    Vec3 new_position = position + velocity * dt;
    double x = new_position.getX();
    double y = new_position.getY();
    double z = new_position.getZ();
    while (x < 0) x += period;
    while (x > period) x -= period;
    while (y < 0) y += period;
    while (y > period) y -= period;
    while (z < 0) z += period;
    while (z > period) z -= period;
    return Vec3(x, y, z);
}

int main(){
    Vec3 r1(2,1,1),r2(9,2,1);
    Vec3 r = AltPeriodicDifferece(r1,r2,100);
    std::cout << r.getX() << " " << r.getY() << " " << r.getZ() << std::endl;
    Vec3 new_position = PeriodicPositionUpdate(r1,-1*r2,1,100);
    std::cout << new_position.getX() << " " << new_position.getY() << " " << new_position.getZ() << std::endl;
    return 0;
};