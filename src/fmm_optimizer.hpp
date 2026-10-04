#ifndef FMM_OPTIMIZER_HPP
#define FMM_OPTIMIZER_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>
#include "common.h"
#include "wind_field.hpp"
#include "itkImage.h"

using SDFImageType = itk::Image<float, 3>;

struct FMMSearchData {
    std::vector<float> T;
    int nx = 0, ny = 0, nz = 0;
    double bx_min = 0.0, by_min = 0.0, bz_min = 0.0, ds = 0.0;
};

namespace fmm_detail {
template <typename TimeoutPredicate>
inline bool compute_arrival_field(
    int nx, int ny, int nz, double ds, const std::vector<float>& speed,
    int goal_x, int goal_y, int goal_z, std::vector<float>& arrival,
    TimeoutPredicate timed_out) {
    const size_t count = static_cast<size_t>(nx) * ny * nz;
    auto index = [=](int x, int y, int z) {
        return static_cast<size_t>((z * ny + y) * nx + x);
    };
    arrival.assign(count, std::numeric_limits<float>::infinity());
    std::vector<unsigned char> state(count, 0);
    auto solve_eikonal = [&](int x, int y, int z) {
        const float inf = std::numeric_limits<float>::infinity();
        float axes[3] = {inf, inf, inf};
        auto accept = [&](int axis, int xx, int yy, int zz) {
            const size_t cell = index(xx, yy, zz);
            if (state[cell] == 2) axes[axis] = std::min(axes[axis], arrival[cell]);
        };
        if (x > 0) accept(0, x - 1, y, z); if (x < nx - 1) accept(0, x + 1, y, z);
        if (y > 0) accept(1, x, y - 1, z); if (y < ny - 1) accept(1, x, y + 1, z);
        if (z > 0) accept(2, x, y, z - 1); if (z < nz - 1) accept(2, x, y, z + 1);
        std::sort(axes, axes + 3);
        int known = 0; while (known < 3 && std::isfinite(axes[known])) ++known;
        if (known == 0 || speed[index(x, y, z)] < 1e-6f) return inf;
        const float tau2 = static_cast<float>(ds * ds) /
            (speed[index(x, y, z)] * speed[index(x, y, z)]);
        float sum = axes[0], sumsq = axes[0] * axes[0], value = axes[0] + std::sqrt(tau2);
        for (int k = 2; k <= known; ++k) {
            const float next = axes[k - 1]; if (value <= next) break;
            sum += next; sumsq += next * next;
            const float disc = sum * sum - static_cast<float>(k) * (sumsq - tau2);
            if (disc < 0.0f) break;
            value = (sum + std::sqrt(disc)) / static_cast<float>(k);
        }
        return std::max(value, axes[0]);
    };
    using Item = std::pair<float, size_t>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
    const size_t goal = index(goal_x, goal_y, goal_z);
    arrival[goal] = 0.0f; state[goal] = 1; queue.push({0.0f, goal});
    size_t expanded = 0;
    while (!queue.empty()) {
        if ((expanded++ & 0xffffu) == 0 && timed_out()) return false;
        const auto [value, cell] = queue.top(); queue.pop();
        if (state[cell] == 2 || value > arrival[cell]) continue;
        state[cell] = 2;
        const int x = static_cast<int>(cell % nx);
        const int y = static_cast<int>((cell / nx) % ny);
        const int z = static_cast<int>(cell / (static_cast<size_t>(nx) * ny));
        const int neighbors[6][3] = {
            {x-1,y,z},{x+1,y,z},{x,y-1,z},{x,y+1,z},{x,y,z-1},{x,y,z+1}
        };
        for (const auto& neighbor : neighbors) {
            if (neighbor[0] < 0 || neighbor[0] >= nx || neighbor[1] < 0 || neighbor[1] >= ny ||
                neighbor[2] < 0 || neighbor[2] >= nz) continue;
            const size_t next = index(neighbor[0], neighbor[1], neighbor[2]);
            if (state[next] == 2 || speed[next] < 1e-4f) continue;
            const float candidate = solve_eikonal(neighbor[0], neighbor[1], neighbor[2]);
            if (candidate < arrival[next]) {
                arrival[next] = candidate; state[next] = 1; queue.push({candidate, next});
            }
        }
    }
    return true;
}
}  // namespace fmm_detail

inline std::vector<Node> plan_fmm_trajectory(
    const Node& start_pt, const Node& goal_pt, SDFImageType::Pointer sdf_image,
    const DEMData& dem, [[maybe_unused]] int max_iters = 300,
    [[maybe_unused]] int num_controls = 0, double safety_margin = 5.0,
    [[maybe_unused]] double learning_rate = 0.05,
    FMMSearchData* out_data = nullptr, const WindField* wind = nullptr,
    double timeout_seconds = 60.0, bool field_only = false) {
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    auto timed_out = [&] {
        return std::chrono::duration<double>(Clock::now() - started).count() >= timeout_seconds;
    };
    std::cout << "[FMM] 启动标准 Fast Marching 路径规划...\n";

    const auto region = sdf_image->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const auto origin = sdf_image->GetOrigin();
    const auto spacing = sdf_image->GetSpacing();
    const double sdf_x_max = origin[0] + (size[0] - 1) * spacing[0];
    const double sdf_y_max = origin[1] + (size[1] - 1) * spacing[1];
    const double sdf_z_max = origin[2] + (size[2] - 1) * spacing[2];
    Node start = start_pt, goal = goal_pt;
    const double distance = get_distance(start, goal);
    const double ds = std::clamp(distance / 250.0, 5.0, 15.0);
    const double pad = 60.0;
    const double bx_min = field_only ? origin[0]
        : std::max(std::min(start.x, goal.x) - pad, origin[0]);
    const double bx_max = field_only ? sdf_x_max
        : std::min(std::max(start.x, goal.x) + pad, sdf_x_max);
    const double by_min = field_only ? origin[1]
        : std::max(std::min(start.y, goal.y) - pad, origin[1]);
    const double by_max = field_only ? sdf_y_max
        : std::min(std::max(start.y, goal.y) + pad, sdf_y_max);
    const double bz_min = field_only ? origin[2]
        : std::max(std::min(start.z, goal.z) - pad, origin[2]);
    const double bz_max = field_only ? sdf_z_max
        : std::min(std::max(start.z, goal.z) + pad, sdf_z_max);
    const int endpoint = field_only ? 1 : 0;
    const int nx = std::max(2, static_cast<int>(std::ceil((bx_max - bx_min) / ds)) + endpoint);
    const int ny = std::max(2, static_cast<int>(std::ceil((by_max - by_min) / ds)) + endpoint);
    const int nz = std::max(2, static_cast<int>(std::ceil((bz_max - bz_min) / ds)) + endpoint);
    const size_t count = static_cast<size_t>(nx) * ny * nz;
    auto index = [=](int x, int y, int z) { return static_cast<size_t>((z * ny + y) * nx + x); };
    auto grid_to_node = [=](int x, int y, int z) {
        return Node(bx_min + x * ds, by_min + y * ds, bz_min + z * ds);
    };
    auto query_esdf = [&](const Node& p) {
        SDFImageType::PointType point; point[0]=p.x; point[1]=p.y; point[2]=p.z;
        SDFImageType::IndexType idx;
        if (sdf_image->TransformPhysicalPointToIndex(point, idx) &&
            sdf_image->GetBufferedRegion().IsInside(idx)) return static_cast<double>(sdf_image->GetPixel(idx));
        return -100.0;
    };
    auto speed_at = [&](const Node& p) {
        const double clearance = query_esdf(p);
        if (!std::isfinite(clearance) || clearance < 0.0) return 1e-6;
        double speed = std::max(std::min(clearance / safety_margin, 1.0), 1e-3);
        if (wind) {
            double u, v; wind->query(p.x, p.y, p.z, dem, u, v);
            speed /= 1.0 + wind->lambda() * std::sqrt(u*u + v*v);
        }
        return std::max(speed, 1e-6);
    };
    auto to_grid = [&](const Node& p, int& x, int& y, int& z) {
        x = std::clamp(static_cast<int>(std::round((p.x-bx_min)/ds)), 0, nx-1);
        y = std::clamp(static_cast<int>(std::round((p.y-by_min)/ds)), 0, ny-1);
        z = std::clamp(static_cast<int>(std::round((p.z-bz_min)/ds)), 0, nz-1);
    };
    int sx,sy,sz,gx,gy,gz; to_grid(start,sx,sy,sz); to_grid(goal,gx,gy,gz);
    std::vector<float> speed(count), T;
    for (int z=0; z<nz; ++z) {
        if (timed_out()) return {};
        for (int y=0; y<ny; ++y) for (int x=0; x<nx; ++x)
            speed[index(x,y,z)] = static_cast<float>(speed_at(grid_to_node(x,y,z)));
    }
    if (speed[index(sx,sy,sz)] < 1e-4f || speed[index(gx,gy,gz)] < 1e-4f) return {};

    if (!fmm_detail::compute_arrival_field(nx, ny, nz, ds, speed, gx, gy, gz, T, timed_out)) return {};
    if (out_data) {
        out_data->T = T;
        out_data->nx = nx;
        out_data->ny = ny;
        out_data->nz = nz;
        out_data->bx_min = bx_min;
        out_data->by_min = by_min;
        out_data->bz_min = bz_min;
        out_data->ds = ds;
    }
    if (field_only) return {};
    if(!std::isfinite(T[index(sx,sy,sz)])) return {};
    std::vector<Node> path{grid_to_node(sx,sy,sz)};
    int cx=sx,cy=sy,cz=sz;
    for(size_t guard=0; !(cx==gx&&cy==gy&&cz==gz) && guard<count; ++guard){
        int bx=cx,by=cy,bz=cz; float best=T[index(cx,cy,cz)]; double best_goal=std::numeric_limits<double>::infinity();
        for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){
            if(dx==0&&dy==0&&dz==0)continue; const int x=cx+dx,y=cy+dy,z=cz+dz;
            if(x<0||x>=nx||y<0||y>=ny||z<0||z>=nz)continue;
            const float value=T[index(x,y,z)]; const double gd=(x-gx)*(x-gx)+(y-gy)*(y-gy)+(z-gz)*(z-gz);
            if(std::isfinite(value)&&value<best-1e-6f){best=value;bx=x;by=y;bz=z;best_goal=gd;}
            else if(value==best&&gd<best_goal){bx=x;by=y;bz=z;best_goal=gd;}
        }
        if(bx==cx&&by==cy&&bz==cz)return {};
        cx=bx;cy=by;cz=bz;path.push_back(grid_to_node(cx,cy,cz));
    }
    if(!(cx==gx&&cy==gy&&cz==gz)) return {};
    path.front()=start; path.push_back(goal);
    return path;
}

#endif
