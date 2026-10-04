#pragma once
#include "dem_data.hpp"
#include "common.h"
#include <cmath>

inline bool is_state_valid_dem(double x, double y, double z,
                               const DEMData& dem, double safety_margin) {
    double dem_z = dem.getElevation(x, y);
    return (z >= dem_z + safety_margin);
}

inline bool is_segment_valid_dem(const Node& n1, const Node& n2,
                                 const DEMData& dem, double step_size,
                                 double safety_margin) {
    double dist = std::sqrt((n1.x-n2.x)*(n1.x-n2.x) +
                            (n1.y-n2.y)*(n1.y-n2.y) +
                            (n1.z-n2.z)*(n1.z-n2.z));
    int steps = (int)std::ceil(dist / step_size);
    for (int i = 1; i <= steps; ++i) {
        double t = (double)i / steps;
        double cx = n1.x + t * (n2.x - n1.x);
        double cy = n1.y + t * (n2.y - n1.y);
        double cz = n1.z + t * (n2.z - n1.z);
        if (!is_state_valid_dem(cx, cy, cz, dem, safety_margin))
            return false;
    }
    return true;
}