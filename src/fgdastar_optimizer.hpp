#ifndef FGDASTAR_OPTIMIZER_HPP
#define FGDASTAR_OPTIMIZER_HPP

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <string>
#include <vector>
#include "common.h"
#include "fmm_optimizer.hpp"
#include "wind_field.hpp"
#include "itkImage.h"

using FGDAStarSDFImageType = itk::Image<float, 3>;

struct FGDAStarConfig {
    double curvature_weight = 0.1;
    double traversal_weight = 1.0;
    double clearance_weight = 0.2;
    double heuristic_weight = 1.0;
    double max_turn_angle_deg = 60.0;
    int corridor_radius = 5;
    bool position_state = false;
    double timeout_seconds = 60.0;
};

struct FGDAStarSearchData {
    FMMSearchData fmm;
    std::vector<Node> guidance_path;
    std::vector<float> guidance_arrival;
    std::vector<unsigned char> corridor;
    std::vector<double> node_x, node_y, node_z;
    std::vector<int> parent_idx;
};

inline std::vector<Node> plan_fgdastar_trajectory(
    const Node& start_pt, const Node& goal_pt,
    FGDAStarSDFImageType::Pointer sdf_image, const DEMData& dem,
    double safety_margin = 5.0, const FGDAStarConfig& config = {},
    const WindField* wind = nullptr, FGDAStarSearchData* out_data = nullptr,
    std::string* error = nullptr) {
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    auto timed_out = [&] {
        return std::chrono::duration<double>(Clock::now() - started).count() >= config.timeout_seconds;
    };
    if (out_data) *out_data = FGDAStarSearchData{};
    auto fail = [&](const std::string& reason) {
        if (error) *error = reason;
        return std::vector<Node>{};
    };

    Node start = start_pt, goal = goal_pt;
    const auto region = sdf_image->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const auto origin = sdf_image->GetOrigin();
    const auto spacing = sdf_image->GetSpacing();
    const double x_max = origin[0] + size[0] * spacing[0];
    const double y_max = origin[1] + size[1] * spacing[1];
    const double z_max = origin[2] + size[2] * spacing[2];
    const double ds = std::clamp(get_distance(start, goal) / 250.0, 5.0, 15.0);
    const double pad = 60.0;
    const double bx_min = std::max(std::min(start.x, goal.x) - pad, origin[0]);
    const double bx_max = std::min(std::max(start.x, goal.x) + pad, x_max);
    const double by_min = std::max(std::min(start.y, goal.y) - pad, origin[1]);
    const double by_max = std::min(std::max(start.y, goal.y) + pad, y_max);
    const double bz_min = std::max(std::min(start.z, goal.z) - pad, origin[2]);
    const double bz_max = std::min(std::max(start.z, goal.z) + pad, z_max);
    const int nx = std::max(2, static_cast<int>(std::ceil((bx_max - bx_min) / ds)));
    const int ny = std::max(2, static_cast<int>(std::ceil((by_max - by_min) / ds)));
    const int nz = std::max(2, static_cast<int>(std::ceil((bz_max - bz_min) / ds)));
    const size_t cell_count = static_cast<size_t>(nx) * ny * nz;
    auto cell = [=](int x, int y, int z) { return static_cast<size_t>((z * ny + y) * nx + x); };
    auto coordinates = [=](size_t c, int& x, int& y, int& z) {
        x = static_cast<int>(c % nx);
        y = static_cast<int>((c / nx) % ny);
        z = static_cast<int>(c / (static_cast<size_t>(nx) * ny));
    };
    auto point = [=](int x, int y, int z) {
        return Node(bx_min + x * ds, by_min + y * ds, bz_min + z * ds);
    };
    auto to_grid = [&](const Node& p, int& x, int& y, int& z) {
        x = std::clamp(static_cast<int>(std::round((p.x - bx_min) / ds)), 0, nx - 1);
        y = std::clamp(static_cast<int>(std::round((p.y - by_min) / ds)), 0, ny - 1);
        z = std::clamp(static_cast<int>(std::round((p.z - bz_min) / ds)), 0, nz - 1);
    };
    auto clearance = [&](const Node& p) {
        FGDAStarSDFImageType::PointType physical;
        physical[0] = p.x; physical[1] = p.y; physical[2] = p.z;
        FGDAStarSDFImageType::IndexType image_index;
        if (sdf_image->TransformPhysicalPointToIndex(physical, image_index) &&
            sdf_image->GetBufferedRegion().IsInside(image_index))
            return static_cast<double>(sdf_image->GetPixel(image_index));
        return -100.0;
    };
    auto speed_at = [&](const Node& p) {
        const double d = clearance(p);
        if (!std::isfinite(d) || d < 0.0) return 1e-6;
        double speed = std::max(std::min(d / safety_margin, 1.0), 1e-3);
        if (wind) {
            double u, v; wind->query(p.x, p.y, p.z, dem, u, v);
            speed /= 1.0 + wind->lambda() * std::sqrt(u * u + v * v);
        }
        return std::max(speed, 1e-6);
    };

    int sx, sy, sz, gx, gy, gz;
    to_grid(start, sx, sy, sz); to_grid(goal, gx, gy, gz);
    const size_t start_cell = cell(sx, sy, sz), goal_cell = cell(gx, gy, gz);
    std::vector<float> speed(cell_count), arrival;
    for (int z = 0; z < nz; ++z) {
        if (timed_out())
            return fail("构建速度场超时: 超过 " + std::to_string(config.timeout_seconds) + "s 限制");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                speed[cell(x, y, z)] = static_cast<float>(speed_at(point(x, y, z)));
    }
    if (speed[start_cell] < 1e-4f || speed[goal_cell] < 1e-4f) {
        return fail("起点/终点落在障碍内或净空不足: 起点ESDF=" +
            std::to_string(clearance(start)) + "m, 终点ESDF=" +
            std::to_string(clearance(goal)) + "m, 安全边距=" +
            std::to_string(safety_margin) + "m");
    }

    if (!fmm_detail::compute_arrival_field(nx, ny, nz, ds, speed, gx, gy, gz, arrival, timed_out))
        return fail("FMM 引导场计算失败或超时");
    if (!std::isfinite(arrival[start_cell]))
        return fail("FMM 引导场中起点不可达: 终点所在自由空间与起点不连通");
    if (out_data) {
        out_data->fmm.T = arrival;
        out_data->fmm.nx = nx; out_data->fmm.ny = ny; out_data->fmm.nz = nz;
        out_data->fmm.bx_min = bx_min; out_data->fmm.by_min = by_min;
        out_data->fmm.bz_min = bz_min; out_data->fmm.ds = ds;
    }

    std::vector<std::array<int, 3>> directions;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx)
                if (dx != 0 || dy != 0 || dz != 0) directions.push_back({dx, dy, dz});
    constexpr int no_direction = 26;

    std::vector<unsigned char> corridor(cell_count, config.corridor_radius <= 0 ? 1 : 0);
    if (config.corridor_radius > 0) {
        int x = sx, y = sy, z = sz;
        for (size_t guard = 0; guard < cell_count; ++guard) {
            if (out_data) {
                out_data->guidance_path.push_back(point(x, y, z));
                out_data->guidance_arrival.push_back(arrival[cell(x, y, z)]);
            }
            for (int oz = -config.corridor_radius; oz <= config.corridor_radius; ++oz)
                for (int oy = -config.corridor_radius; oy <= config.corridor_radius; ++oy)
                    for (int ox = -config.corridor_radius; ox <= config.corridor_radius; ++ox) {
                        if (ox * ox + oy * oy + oz * oz > config.corridor_radius * config.corridor_radius) continue;
                        const int xx = x + ox, yy = y + oy, zz = z + oz;
                        if (xx >= 0 && xx < nx && yy >= 0 && yy < ny && zz >= 0 && zz < nz)
                            corridor[cell(xx, yy, zz)] = 1;
                    }
            if (x == gx && y == gy && z == gz) break;
            float best = arrival[cell(x, y, z)]; int bx = x, by = y, bz = z;
            for (const auto& d : directions) {
                const int xx = x + d[0], yy = y + d[1], zz = z + d[2];
                if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
                if (arrival[cell(xx, yy, zz)] < best - 1e-6f) {
                    best = arrival[cell(xx, yy, zz)]; bx = xx; by = yy; bz = zz;
                }
            }
            if (bx == x && by == y && bz == z)
                return fail("走廊构建失败: FMM 参考路径回溯在格(" +
                    std::to_string(x) + "," + std::to_string(y) + "," +
                    std::to_string(z) + ")处无更低到达时间邻居");
            x = bx; y = by; z = bz;
        }
        corridor[start_cell] = corridor[goal_cell] = 1;
    }
    if (out_data) out_data->corridor = corridor;

    auto edge_cost = [&](const Node& a, const Node& b, double& traversal, double& risk) {
        const double length = get_distance(a, b);
        const double sample_step = std::max(0.5, std::min(ds * 0.25, safety_margin * 0.5));
        const int samples = std::max(1, static_cast<int>(std::ceil(length / sample_step)));
        risk = 0.0;
        for (int k = 0; k <= samples; ++k) {
            const double t = static_cast<double>(k) / samples;
            Node p(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), a.z + t * (b.z - a.z));
            const double d = clearance(p);
            if (!std::isfinite(d) || d < safety_margin) return false;
            risk += 1.0 / std::max(d, 0.1);
        }
        risk = length * risk / static_cast<double>(samples + 1);
        traversal = length;
        if (wind) {
            Node midpoint((a.x + b.x) * 0.5, (a.y + b.y) * 0.5, (a.z + b.z) * 0.5);
            double u, v; wind->query(midpoint.x, midpoint.y, midpoint.z, dem, u, v);
            const double horizontal = std::hypot(b.x - a.x, b.y - a.y);
            if (horizontal > 1e-9) {
                const double headwind = -(u * (b.x - a.x) + v * (b.y - a.y)) / horizontal;
                traversal *= 1.0 + wind->lambda() * std::max(0.0, headwind);
            }
        }
        return true;
    };
    auto heuristic = [&](size_t c) {
        return std::isfinite(arrival[c]) ? static_cast<double>(arrival[c])
                                         : std::numeric_limits<double>::infinity();
    };
    if (start_cell == goal_cell) {
        double traversal = 0.0, risk = 0.0;
        if (!edge_cost(start, goal, traversal, risk))
            return fail("起终点位于同一网格但精确连接不满足安全边距");
        return get_distance(start, goal) > 1e-9
            ? std::vector<Node>{start, goal}
            : std::vector<Node>{start};
    }

    // 仅为走廊内单元分配有向边缓存。同一栅格边会被不同入射方向
    // 重复访问，缓存后每条边最多执行一次 ESDF 采样和风场查询。
    const size_t no_corridor_index = std::numeric_limits<size_t>::max();
    std::vector<size_t> corridor_index(cell_count, no_corridor_index);
    std::vector<size_t> corridor_cells;
    corridor_cells.reserve(cell_count);
    for (size_t c = 0; c < cell_count; ++c) {
        if (corridor[c]) {
            corridor_index[c] = corridor_cells.size();
            corridor_cells.push_back(c);
        }
    }
    const size_t corridor_cell_count = corridor_cells.size();
    const size_t cached_edge_count = corridor_cell_count * directions.size();
    std::vector<unsigned char> edge_state(cached_edge_count, 0); // 0未知, 1无效, 2有效
    std::vector<double> edge_traversal(cached_edge_count, 0.0);
    std::vector<double> edge_risk(cached_edge_count, 0.0);

    // 26 个方向只有 676 种转移，预先缓存转角可行性和曲率代价，
    // 避免搜索内层循环重复执行 sqrt/acos。
    std::array<std::array<unsigned char, 26>, 27> transition_allowed{};
    std::array<std::array<double, 26>, 27> transition_curvature{};
    const double pi = std::acos(-1.0);
    for (int next_direction = 0; next_direction < 26; ++next_direction)
        transition_allowed[no_direction][next_direction] = 1;
    for (int previous_direction = 0; previous_direction < 26; ++previous_direction) {
        const auto& previous = directions[previous_direction];
        const double previous_length = ds * std::sqrt(static_cast<double>(
            previous[0] * previous[0] + previous[1] * previous[1] + previous[2] * previous[2]));
        for (int next_direction = 0; next_direction < 26; ++next_direction) {
            const auto& next = directions[next_direction];
            const double next_length = ds * std::sqrt(static_cast<double>(
                next[0] * next[0] + next[1] * next[1] + next[2] * next[2]));
            const double cosine = std::clamp(
                (previous[0] * next[0] + previous[1] * next[1] + previous[2] * next[2]) /
                    (previous_length * next_length / (ds * ds)),
                -1.0, 1.0);
            const double angle = std::acos(cosine);
            transition_allowed[previous_direction][next_direction] =
                config.max_turn_angle_deg <= 0.0 || angle * 180.0 / pi <= config.max_turn_angle_deg;
            const double support = 0.5 * (previous_length + next_length);
            transition_curvature[previous_direction][next_direction] =
                angle * angle / std::max(support, 1e-9);
        }
    }

    // 方向状态仅为走廊单元分配，避免走廊外体素占用 27 份状态内存。
    const int labels_per_cell = config.position_state ? 1 : 27;
    const size_t label_count = corridor_cell_count * static_cast<size_t>(labels_per_cell);
    auto label = [&](size_t global_cell, int direction) {
        const size_t compact_cell = corridor_index[global_cell];
        return compact_cell * labels_per_cell + (config.position_state ? 0 : direction);
    };
    auto label_cell = [&](size_t l) {
        return corridor_cells[l / labels_per_cell];
    };
    auto label_direction = [&](size_t l) {
        return config.position_state ? no_direction : static_cast<int>(l % labels_per_cell);
    };
    std::vector<double> g_score(label_count, std::numeric_limits<double>::infinity());
    std::vector<size_t> parent(label_count, std::numeric_limits<size_t>::max());
    std::vector<int> search_index;
    if (out_data) search_index.assign(label_count, -1);
    struct OpenItem { double f; size_t label; };
    struct Greater { bool operator()(const OpenItem& a, const OpenItem& b) const { return a.f > b.f; } };
    std::priority_queue<OpenItem, std::vector<OpenItem>, Greater> open;
    const size_t start_label = label(start_cell, no_direction);
    g_score[start_label] = 0.0;
    if (out_data) {
        out_data->node_x.push_back(start.x); out_data->node_y.push_back(start.y);
        out_data->node_z.push_back(start.z); out_data->parent_idx.push_back(-1);
        search_index[start_label] = 0;
    }
    open.push({config.heuristic_weight * heuristic(start_cell), start_label});
    size_t goal_label = std::numeric_limits<size_t>::max(), expanded = 0;
    while (!open.empty()) {
        if ((expanded++ & 0x3fffu) == 0 && timed_out())
            return fail("方向状态 A* 搜索超时: 已扩展 " +
                std::to_string(expanded) + " 个节点, 超过 " +
                std::to_string(config.timeout_seconds) + "s 限制");
        const OpenItem item = open.top(); open.pop();
        const size_t current_cell = label_cell(item.label);
        const double expected = g_score[item.label] + config.heuristic_weight * heuristic(current_cell);
        if (item.f > expected + 1e-9) continue;
        if (current_cell == goal_cell) { goal_label = item.label; break; }
        int x, y, z; coordinates(current_cell, x, y, z);
        const int previous_direction = label_direction(item.label);
        const Node current_point = item.label == start_label ? start : point(x, y, z);
        for (int direction = 0; direction < 26; ++direction) {
            const auto& d = directions[direction];
            const int xx = x + d[0], yy = y + d[1], zz = z + d[2];
            if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || zz < 0 || zz >= nz) continue;
            const size_t next_cell = cell(xx, yy, zz);
            if (!corridor[next_cell] || speed[next_cell] < 1e-4f) continue;
            // 首尾边直接使用精确航点，确保搜索代价、碰撞检查和最终统计使用同一几何路径。
            const Node next_point = next_cell == goal_cell ? goal : point(xx, yy, zz);

            double curvature_cost = transition_curvature[previous_direction][direction];
            const bool endpoint_transition =
                !config.position_state && previous_direction != no_direction &&
                (parent[item.label] == start_label || next_cell == goal_cell);
            if (!config.position_state && !endpoint_transition &&
                !transition_allowed[previous_direction][direction]) continue;
            if (endpoint_transition) {
                Node previous_point = start;
                if (parent[item.label] != start_label) {
                    int px, py, pz;
                    coordinates(label_cell(parent[item.label]), px, py, pz);
                    previous_point = point(px, py, pz);
                }
                const double ax = current_point.x - previous_point.x;
                const double ay = current_point.y - previous_point.y;
                const double az = current_point.z - previous_point.z;
                const double bx = next_point.x - current_point.x;
                const double by = next_point.y - current_point.y;
                const double bz = next_point.z - current_point.z;
                const double a_length = std::sqrt(ax * ax + ay * ay + az * az);
                const double b_length = std::sqrt(bx * bx + by * by + bz * bz);
                if (a_length < 1e-9 || b_length < 1e-9) continue;
                const double angle = std::acos(std::clamp(
                    (ax * bx + ay * by + az * bz) / (a_length * b_length), -1.0, 1.0));
                if (config.max_turn_angle_deg > 0.0 &&
                    angle * 180.0 / pi > config.max_turn_angle_deg) continue;
                curvature_cost = angle * angle / (0.5 * (a_length + b_length));
            }

            double traversal = 0.0, risk = 0.0;
            if (item.label == start_label || next_cell == goal_cell) {
                if (!edge_cost(current_point, next_point, traversal, risk)) continue;
            } else {
                const size_t source_corridor_index = corridor_index[current_cell];
                const size_t edge_index = source_corridor_index * directions.size() + direction;
                if (edge_state[edge_index] == 0) {
                    if (edge_cost(current_point, next_point, traversal, risk)) {
                        edge_state[edge_index] = 2;
                        edge_traversal[edge_index] = traversal;
                        edge_risk[edge_index] = risk;
                    } else {
                        edge_state[edge_index] = 1;
                    }
                }
                if (edge_state[edge_index] == 1) continue;
                traversal = edge_traversal[edge_index];
                risk = edge_risk[edge_index];
            }
            const size_t next_label = label(next_cell, direction);
            const double candidate = g_score[item.label] +
                config.traversal_weight * traversal + config.clearance_weight * risk +
                config.curvature_weight * curvature_cost;
            if (candidate + 1e-9 < g_score[next_label]) {
                g_score[next_label] = candidate; parent[next_label] = item.label;
                if (out_data) {
                    const int parent_search_index = search_index[item.label];
                    if (search_index[next_label] < 0) {
                        search_index[next_label] = static_cast<int>(out_data->node_x.size());
                        out_data->node_x.push_back(bx_min + xx * ds);
                        out_data->node_y.push_back(by_min + yy * ds);
                        out_data->node_z.push_back(bz_min + zz * ds);
                        out_data->parent_idx.push_back(parent_search_index);
                    } else {
                        out_data->parent_idx[search_index[next_label]] = parent_search_index;
                    }
                }
                open.push({candidate + config.heuristic_weight * heuristic(next_cell), next_label});
            }
        }
    }
    if (goal_label == std::numeric_limits<size_t>::max()) {
        return fail("搜索耗尽仍未到达终点: 走廊半径=" +
            std::to_string(config.corridor_radius) + "格, 最大转角=" +
            (config.max_turn_angle_deg > 0.0
                 ? std::to_string(config.max_turn_angle_deg) + "°"
                 : "无") +
            ", 已扩展 " + std::to_string(expanded) +
            " 个节点 — 走廊内可能不存在满足转角约束的可行路径, "
            "可尝试增大 --fgda-corridor-radius 或放宽 --fgda-max-turn-angle");
    }
    std::vector<Node> path;
    for (size_t current = goal_label; current != std::numeric_limits<size_t>::max(); current = parent[current]) {
        int x, y, z; coordinates(label_cell(current), x, y, z);
        path.push_back(point(x, y, z));
        if (current == start_label) break;
    }
    if (path.empty() || label_cell(start_label) != start_cell)
        return fail("路径回溯异常: 未能从终点标签回溯到起始标签");
    std::reverse(path.begin(), path.end());
    // 搜索首尾边已使用精确航点并完成安全/转角检查，此处只恢复同一几何端点。
    path.front() = start;
    path.back() = goal;
    return path;
}

#endif
