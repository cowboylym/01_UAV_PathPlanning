#ifndef ASTAR_OPTIMIZER_HPP
#define ASTAR_OPTIMIZER_HPP

// ============================================================================
//  ESDF 轨迹优化器 - A* + ESDF 投影平滑
//
//  架构:
//    前端: 风险感知 A* — 3D 网格上的最短路径搜索 (26 邻居),
//          边代价 = 段长 × 风险权重 (参考 FMM: 风险权重 = 1/F_esdf, F_esdf=min(d/margin,1)),
//          ESDF 既做硬闸门 (碰撞检查), 也作为代价项 (贴障碍加价, 鼓励绕行).
//    后端: ESDF 投影 + 3 点平滑 — 把路径点推到安全距离之外,
//          再用加权平均平滑路径, 每轮平滑后重新投影防止碰撞.
// ============================================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "common.h"
#include "dem_data.hpp"
#include "wind_field.hpp"                 // [新增] 风场代价 (节能规划)
#include "itkImage.h"

using SDFImageType = itk::Image<float, 3>;

// A* 搜索过程数据 (用于可视化已访问节点和边)
struct AStarSearchData {
    std::vector<double> node_x, node_y, node_z;  // 所有已发现节点坐标
    std::vector<int>    parent_idx;              // 父节点索引 (-1=根)
};

// ============================================================================
//  A* + ESDF 投影平滑主函数
//  函数签名保持与原优化器一致, 以兼容 main.cpp 调用
//    max_iters     : (保留兼容, 未使用)
//    num_controls  : (保留兼容, 未使用)
//    safety_margin : ESDF 安全距离阈值 (m)
//    learning_rate : (保留兼容, 未使用)
// ============================================================================
inline std::vector<Node> plan_astar_trajectory(
    const Node& start_pt,
    const Node& goal_pt,
    SDFImageType::Pointer sdfImage,
    const DEMData& dem,
    int max_iters = 300,
    int num_controls = 0,
    double safety_margin = 5.0,
    double learning_rate = 0.05,
    AStarSearchData* out_data = nullptr,          // 可选: 导出搜索节点用于可视化
    const WindField* wind = nullptr,
    double timeout_seconds = 60.0,
    bool enable_internal_postprocess = false)
{
    using TimeoutClock = std::chrono::steady_clock;
    const auto planning_started = TimeoutClock::now();
    auto timed_out = [&]() {
        return std::chrono::duration<double>(TimeoutClock::now() - planning_started).count()
            >= timeout_seconds;
    };
    auto log_timeout = [&](const char* phase) {
        std::cerr << "[A*] 失败: " << phase << " 超过 " << timeout_seconds
                  << " 秒，返回空路径\n";
    };

    (void)max_iters;      // 保留参数兼容性
    (void)num_controls;
    (void)learning_rate;
    std::cout << "[A*] 启动 A* + ESDF 投影平滑...\n";

    using Vec3 = Eigen::Vector3d;

    // ==================================================================
    // 1. Setup: ESDF 元数据, 边界框, 参数
    // ==================================================================
    SDFImageType::RegionType region = sdfImage->GetLargestPossibleRegion();
    SDFImageType::SizeType    size   = region.GetSize();
    SDFImageType::PointType   origin = sdfImage->GetOrigin();
    SDFImageType::SpacingType spacing= sdfImage->GetSpacing();

    double sdf_x_min = origin[0], sdf_y_min = origin[1], sdf_z_min = origin[2];
    double sdf_x_max = origin[0] + size[0] * spacing[0];
    double sdf_y_max = origin[1] + size[1] * spacing[1];
    double sdf_z_max = origin[2] + size[2] * spacing[2];

    // ESDF 查询
    auto query_esdf = [&](double x, double y, double z) -> double {
        SDFImageType::PointType pt; pt[0]=x; pt[1]=y; pt[2]=z;
        SDFImageType::IndexType idx;
        if (sdfImage->TransformPhysicalPointToIndex(pt, idx)) {
            if (sdfImage->GetBufferedRegion().IsInside(idx))
                return static_cast<double>(sdfImage->GetPixel(idx));
        }
        return -100.0;
    };

    // ESDF 梯度 (中心差分, 归一化)
    auto esdf_gradient = [&](const Vec3& p) -> Vec3 {
        double e = std::max(spacing[0], std::max(spacing[1], spacing[2])) * 0.5;
        double gx = query_esdf(p.x()+e, p.y(), p.z()) - query_esdf(p.x()-e, p.y(), p.z());
        double gy = query_esdf(p.x(), p.y()+e, p.z()) - query_esdf(p.x(), p.y()-e, p.z());
        double gz = query_esdf(p.x(), p.y(), p.z()+e) - query_esdf(p.x(), p.y(), p.z()-e);
        Vec3 g(gx, gy, gz);
        double n = g.norm();
        if (n < 1e-9) return Vec3(0, 0, 1);  // 梯度退化: 向上脱困
        return g / n;
    };

    Vec3 start_pos(start_pt.x, start_pt.y, start_pt.z);
    Vec3 goal_pos(goal_pt.x, goal_pt.y, goal_pt.z);
    double sg_dist = (goal_pos - start_pos).norm();

    // A* 搜索网格 (普通 A*: 网格即搜索空间, 网格步长 = 邻居步长)
    const double pad = 60.0;
    double bx_min = std::max(std::min(start_pos.x(), goal_pos.x()) - pad, sdf_x_min);
    double bx_max = std::min(std::max(start_pos.x(), goal_pos.x()) + pad, sdf_x_max);
    double by_min = std::max(std::min(start_pos.y(), goal_pos.y()) - pad, sdf_y_min);
    double by_max = std::min(std::max(start_pos.y(), goal_pos.y()) + pad, sdf_y_max);
    double bz_min = std::max(std::min(start_pos.z(), goal_pos.z()) - pad, sdf_z_min);
    double bz_max = std::min(std::max(start_pos.z(), goal_pos.z()) + pad, sdf_z_max);

    const double ds = std::clamp(sg_dist / 250.0, 5.0, 15.0);  // A* 步长 (m, 与 FMM 统一)
    int nx = std::max(2, (int)std::ceil((bx_max - bx_min) / ds));
    int ny = std::max(2, (int)std::ceil((by_max - by_min) / ds));
    int nz = std::max(2, (int)std::ceil((bz_max - bz_min) / ds));
    size_t total_cells = (size_t)nx * ny * nz;

    std::cout << "[A*] 起终点距离 " << sg_dist << " m, A* 网格 "
              << nx << "x" << ny << "x" << nz << " (ds=" << ds << " m)\n";

    auto phys_to_cell = [&](const Vec3& p, int& cx, int& cy, int& cz) {
        cx = std::clamp((int)((p.x() - bx_min) / ds), 0, nx - 1);
        cy = std::clamp((int)((p.y() - by_min) / ds), 0, ny - 1);
        cz = std::clamp((int)((p.z() - bz_min) / ds), 0, nz - 1);
    };
    auto cell_idx = [&](int x, int y, int z) -> size_t {
        return (size_t)((z * ny + y) * nx + x);
    };

    // A* 宽松净空阈值: 前端只求可行路径, 完整 safety_margin 由后处理投影保证.
    const double astar_clearance = std::min(safety_margin * 0.5, 5.0);
    std::cout << "[A*] A* 净空阈值: " << astar_clearance
              << " m (后处理投影保证完整 " << safety_margin << " m)\n";

    // 检查起点/终点是否在障碍内 (用调整后的 z 和宽松阈值)
    if (query_esdf(start_pos.x(), start_pos.y(), start_pos.z()) < astar_clearance) {
        std::cerr << "[A*] 错误: start 在障碍物内或越界 (SDF="
                  << query_esdf(start_pos.x(), start_pos.y(), start_pos.z())
                  << " < " << astar_clearance << ")!\n";
        return {};
    }
    if (query_esdf(goal_pos.x(), goal_pos.y(), goal_pos.z()) < astar_clearance) {
        std::cerr << "[A*] 错误: goal 在障碍物内或越界 (SDF="
                  << query_esdf(goal_pos.x(), goal_pos.y(), goal_pos.z())
                  << " < " << astar_clearance << ")!\n";
        return {};
    }

    // ==================================================================
    // 2. 前端: 风险感知 A*
    //    状态 = 位置, 邻居 = 26 方向网格步进, 启发式 = 欧氏距离 (admissible),
    //    边代价 = 段长 × 风险权重 (参考 FMM 速度场: 贴障碍加价, 鼓励绕行).
    // ==================================================================
    struct AStarNode {
        Vec3 pos;
        double g, f;
        int parent;
    };

    // 26 邻居方向 (3D 网格)
    std::vector<Vec3> neighbor_offsets;
    for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                neighbor_offsets.push_back(Vec3(dx, dy, dz) * ds);
            }

    auto heuristic = [&](const Vec3& pos) -> double {
        return (goal_pos - pos).norm();
    };

    std::vector<AStarNode> nodes;
    nodes.reserve(50000);
    std::vector<double> visited(total_cells,
                                std::numeric_limits<double>::infinity());

    using PQItem = std::pair<double, int>;  // (f, node_idx)
    std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> open;

    // 起点
    nodes.push_back({start_pos, 0.0, heuristic(start_pos), -1});
    open.push({nodes[0].f, 0});
    {
        int cx, cy, cz; phys_to_cell(start_pos, cx, cy, cz);
        visited[cell_idx(cx, cy, cz)] = 0.0;
    }

    const double goal_thresh = ds * 1.5;
    const size_t MAX_NODES = 500000;
    int goal_node = -1;

    while (!open.empty() && nodes.size() < MAX_NODES) {
        if (timed_out()) {
            log_timeout("A* 搜索");
            return {};
        }
        auto [fval, idx] = open.top(); open.pop();
        if (fval > nodes[idx].f) continue;  // 过期项

        Vec3 cur_pos = nodes[idx].pos;
        double cur_g = nodes[idx].g;

        // 到达目标
        if ((cur_pos - goal_pos).norm() < goal_thresh) {
            goal_node = idx;
            break;
        }

        // 扩展 26 邻居
        for (const Vec3& offset : neighbor_offsets) {
            Vec3 next_pos = cur_pos + offset;

            // 边界检查
            if (next_pos.x() < bx_min || next_pos.x() > bx_max ||
                next_pos.y() < by_min || next_pos.y() > by_max ||
                next_pos.z() < bz_min || next_pos.z() > bz_max) continue;

            // ESDF 硬闸门: 低于宽松阈值视为碰撞 (完整 safety_margin 由后处理投影保证)
            if (query_esdf(next_pos.x(), next_pos.y(), next_pos.z()) < astar_clearance) continue;

            // 边代价 = 段长 × ESDF 风险权重。
            // d>=margin 时权重为 1；贴近障碍时加价。
            double seg_len = offset.norm();
            Vec3 mid_pos(0.5 * (cur_pos.x() + next_pos.x()),
                         0.5 * (cur_pos.y() + next_pos.y()),
                         0.5 * (cur_pos.z() + next_pos.z()));
            double d_mid = query_esdf(mid_pos.x(), mid_pos.y(), mid_pos.z());
            double risk_weight = std::max(safety_margin / std::max(d_mid, 0.5), 1.0);
            double step_cost = seg_len * risk_weight;

            // [新增] 风场代价: 边代价 ×(1 + λ_w·max(0, 顶风分量))
            //   系数恒 ≥1 → 欧氏启发式保持 admissible; 顺风不奖励.
            if (wind) {
                step_cost *= wind->edgeCostCoeff(
                    cur_pos.x(), cur_pos.y(), cur_pos.z(),
                    next_pos.x(), next_pos.y(), next_pos.z(), dem);
            }
            double next_g = cur_g + step_cost;

            // 网格剪枝
            int cx, cy, cz; phys_to_cell(next_pos, cx, cy, cz);
            size_t cell = cell_idx(cx, cy, cz);
            if (next_g >= visited[cell]) continue;
            visited[cell] = next_g;

            int next_idx = (int)nodes.size();
            nodes.push_back({next_pos, next_g,
                             next_g + heuristic(next_pos), idx});
            open.push({nodes[next_idx].f, next_idx});
        }
    }

    // 回溯路径
    std::vector<Vec3> astar_path;
    if (goal_node >= 0) {
        int idx = goal_node;
        while (idx >= 0) {
            astar_path.push_back(nodes[idx].pos);
            idx = nodes[idx].parent;
        }
        std::reverse(astar_path.begin(), astar_path.end());
        std::cout << "[A*] A* 到达目标, 路径 " << astar_path.size()
                  << " 个节点 (扩展 " << nodes.size() << " 节点)\n";
    } else {
        // A* 未到达: 取离 goal 最近的节点, 追加连接段
        int best = 0;
        double best_d = (nodes[0].pos - goal_pos).norm();
        for (size_t i = 1; i < nodes.size(); ++i) {
            double d = (nodes[i].pos - goal_pos).norm();
            if (d < best_d) { best_d = d; best = (int)i; }
        }
        if (best_d > 100.0) {
            std::cerr << "[A*] 错误: A* 未收敛, 最近点距 goal "
                      << best_d << " m\n";
            return {};
        }
        int idx = best;
        while (idx >= 0) {
            astar_path.push_back(nodes[idx].pos);
            idx = nodes[idx].parent;
        }
        std::reverse(astar_path.begin(), astar_path.end());
        // 追加直线连接段
        Vec3 last = astar_path.back();
        double seg_len = (goal_pos - last).norm();
        int n_conn = std::max(2, (int)std::ceil(seg_len / ds));
        for (int s = 1; s <= n_conn; ++s) {
            double t = (double)s / n_conn;
            astar_path.push_back(last + t * (goal_pos - last));
        }
        std::cout << "[A*] A* 近似到达 (距 goal " << best_d
                  << " m), 追加连接段, 路径 " << astar_path.size() << " 节点\n";
    }

    // 确保首尾精确
    astar_path.front() = start_pos;
    astar_path.back()  = goal_pos;

    // ==================================================================
    // 3. 后处理: 迭代式 ESDF 投影 + 段间加密 + 3 点平滑
    //    修复: 原 5 次单步大跳投影在窄通道中失效 (min_d=6.7m < 10m).
    //          改为 FMM 同款迭代式投影 (≤60步, +z 兜底) + 段内加密.
    // ==================================================================

    // 3a. 单点迭代式 ESDF 投影
    auto project_point_to_clear = [&](Vec3& p) -> bool {
        bool moved = false;
        for (int it = 0; it < 60; ++it) {
            double d = query_esdf(p.x(), p.y(), p.z());
            if (d >= safety_margin) break;
            Vec3 grad = esdf_gradient(p);   // 已归一化, 退化时返回 (0,0,1)
            double push = (safety_margin - d) + 1.0;
            p += grad * push;
            p.x() = std::clamp(p.x(), bx_min, bx_max);
            p.y() = std::clamp(p.y(), by_min, by_max);
            p.z() = std::clamp(p.z(), bz_min, bz_max);
            moved = true;
        }
        return moved;
    };

    // 3b. 路径点投影 + 段内加密 (步长 4m, 检测每段最差点并插入)
    auto enforce_clear_and_densify = [&](std::vector<Vec3>& path) -> int {
        const size_t MAX_POINTS = 1000;
        const double SAMPLE_STEP = 4.0;
        const int MAX_OUTER = 20;
        int total_fixed = 0, total_inserted = 0;
        for (int outer = 0; outer < MAX_OUTER && path.size() < MAX_POINTS; ++outer) {
            int pviol = 0;
            for (auto& p : path) {
                if (project_point_to_clear(p)) { ++pviol; ++total_fixed; }
            }
            std::vector<std::pair<size_t, Vec3>> to_insert;
            for (size_t i = 1; i < path.size() && path.size() + to_insert.size() < MAX_POINTS; ++i) {
                Vec3 dvec = path[i] - path[i-1];
                double dh = std::sqrt(dvec.x()*dvec.x() + dvec.y()*dvec.y());
                if (dh < SAMPLE_STEP) continue;
                int n_samples = (int)(dh / SAMPLE_STEP);
                double worst_t = -1, worst_d = safety_margin;
                for (int s = 1; s < n_samples; ++s) {
                    double t = (double)s / n_samples;
                    Vec3 sp = path[i-1] + t * dvec;
                    double d = query_esdf(sp.x(), sp.y(), sp.z());
                    if (d < worst_d) { worst_d = d; worst_t = t; }
                }
                if (worst_t < 0) continue;
                Vec3 mp = path[i-1] + worst_t * dvec;
                project_point_to_clear(mp);
                to_insert.emplace_back(i, mp);
            }
            for (auto it = to_insert.rbegin(); it != to_insert.rend(); ++it)
                path.insert(path.begin() + it->first, it->second);
            total_inserted += (int)to_insert.size();
            if (pviol == 0 && to_insert.empty()) break;
        }
        std::cout << "[A*] ESDF 净空: 投影 " << total_fixed << " 点, 加密 "
                  << total_inserted << " 点 (最终 " << path.size() << " 点)\n";
        return total_fixed + total_inserted;
    };

    if (enable_internal_postprocess) enforce_clear_and_densify(astar_path);

    // 3c. 旧内部后处理仅供兼容；主实验在 main.cpp 统一处理。
    for (int iter = 0; enable_internal_postprocess && iter < 5; ++iter) {
        std::vector<Vec3> smoothed = astar_path;
        // 内部点: 5 点高斯加权
        for (size_t i = 2; i + 2 < astar_path.size(); ++i) {
            smoothed[i] = (astar_path[i-2] + 4.0*astar_path[i-1] + 6.0*astar_path[i]
                        + 4.0*astar_path[i+1] + astar_path[i+2]) / 16.0;
        }
        // 边界次邻点: 3 点加权 [1,2,1]/4
        if (astar_path.size() >= 3) {
            size_t last = astar_path.size() - 1;
            smoothed[1] = (astar_path[0] + 2.0*astar_path[1] + astar_path[2]) * 0.25;
            smoothed[last-1] = (astar_path[last-2] + 2.0*astar_path[last-1] + astar_path[last]) * 0.25;
        }
        astar_path = smoothed;
        enforce_clear_and_densify(astar_path);
    }

    // 确保首尾精确
    astar_path.front() = start_pos;
    astar_path.back()  = goal_pos;

    std::cout << "[A*] 后处理完成 (ESDF 投影 + 平滑), 路径 "
              << astar_path.size() << " 个点\n";

    // ==================================================================
    // 4. 输出: 转换为 Node + 统计
    // ==================================================================
    std::vector<Node> final_path;
    final_path.reserve(astar_path.size());
    for (const Vec3& p : astar_path)
        final_path.push_back(Node(p.x(), p.y(), p.z()));

    // 统计
    double min_d = std::numeric_limits<double>::max();
    double max_d = std::numeric_limits<double>::lowest();
    double total_3d = 0, total_horiz = 0;

    for (size_t i = 0; i < final_path.size(); ++i) {
        double d = query_esdf(final_path[i].x, final_path[i].y, final_path[i].z);
        min_d = std::min(min_d, d);
        max_d = std::max(max_d, d);
        if (i == 0) continue;

        double dx = final_path[i].x - final_path[i-1].x;
        double dy = final_path[i].y - final_path[i-1].y;
        double dz = final_path[i].z - final_path[i-1].z;
        double dh = std::sqrt(dx*dx + dy*dy);
        double dl = std::sqrt(dx*dx + dy*dy + dz*dz);
        total_3d += dl;
        total_horiz += dh;

    }

    double straight_horiz = std::sqrt(
        (goal_pt.x - start_pt.x)*(goal_pt.x - start_pt.x) +
        (goal_pt.y - start_pt.y)*(goal_pt.y - start_pt.y));

    std::cout << "[A*] ===== 最终路径统计 =====\n";
    std::cout << "  - 路径点数      : " << final_path.size() << "\n";
    std::cout << "  - ESDF 距离区间 : [" << min_d << ", " << max_d << "] m\n";
    std::cout << "  - 3D 路径总长   : " << total_3d << " m\n";
    std::cout << "  - 水平路径总长  : " << total_horiz << " m (直线 "
              << straight_horiz << " m, 延伸 "
              << (total_horiz - straight_horiz) << " m)\n";
    std::cout << "  - 最小安全余量  : " << (min_d >= safety_margin ? "OK" : "WARN") << "\n";

    // 导出搜索节点用于可视化
    if (out_data) {
        out_data->node_x.reserve(nodes.size());
        out_data->node_y.reserve(nodes.size());
        out_data->node_z.reserve(nodes.size());
        out_data->parent_idx.reserve(nodes.size());
        for (const auto& n : nodes) {
            out_data->node_x.push_back(n.pos.x());
            out_data->node_y.push_back(n.pos.y());
            out_data->node_z.push_back(n.pos.z());
            out_data->parent_idx.push_back(n.parent);
        }
    }

    return final_path;
}

#endif // ASTAR_OPTIMIZER_HPP
