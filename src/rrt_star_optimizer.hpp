#ifndef RRT_STAR_OPTIMIZER_HPP
#define RRT_STAR_OPTIMIZER_HPP

// ============================================================================
//  RRT* 轨迹规划器 — 快速探索随机树 (Rapidly-exploring Random Tree Star)
//
//  架构:
//    前端: RRT* — 随机采样 + 最近邻 + rewiring, 渐近最优地探索连续空间
//    后端: ESDF 投影 + 3 点平滑
//
//  特点:
//    - 无需网格, 直接在连续空间采样
//    - 通过 rewiring 渐近逼近最优路径
//    - 边代价纳入风险项 (参考 FMM: 段长 × 风险权重, 贴障碍加价鼓励绕行)
//    - 适合高维空间和复杂障碍场景
//    - goal-biased 采样加速收敛
// ============================================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <limits>
#include <Eigen/Core>
#include "common.h"
#include "dem_data.hpp"
#include "wind_field.hpp"                 // [新增] 风场代价 (节能规划)
#include "itkImage.h"

using SDFImageType = itk::Image<float, 3>;

// RRT* 树节点
struct RRTNode {
    Eigen::Vector3d pos;
    int parent;
    double cost;  // 从起点到该节点的累积代价
};

// RRT* 搜索过程数据 (用于可视化搜索树)
struct RRTSearchData {
    std::vector<double> node_x, node_y, node_z;  // 树节点坐标
    std::vector<int>    parent_idx;              // 父节点索引 (-1=根)
    std::vector<double> cost;                    // 累积代价
};

// ============================================================================
//  RRT* 主函数
//  函数签名与 plan_esdf_trajectory 一致, 便于 main.cpp 切换
// ============================================================================
inline std::vector<Node> plan_rrt_star_trajectory(
    const Node& start_pt,
    const Node& goal_pt,
    SDFImageType::Pointer sdfImage,
    const DEMData& dem,
    int max_iters = 300,
    int num_controls = 0,
    double safety_margin = 5.0,
    double learning_rate = 0.05,
    RRTSearchData* out_data = nullptr,             // 可选: 导出搜索树用于可视化
    const WindField* wind = nullptr,               // [新增] 可选: 风场代价 (null=关闭)
    unsigned int seed = 42,
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
        std::cerr << "[RRT*] 失败: " << phase << " 超过 " << timeout_seconds
                  << " 秒，返回空路径\n";
    };

    (void)num_controls;
    (void)learning_rate;

    std::cout << "[RRT*] 启动 RRT* 路径规划...\n";

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
        if (n < 1e-9) return Vec3(0, 0, 1);
        return g / n;
    };

    Vec3 start_pos(start_pt.x, start_pt.y, start_pt.z);
    Vec3 goal_pos(goal_pt.x, goal_pt.y, goal_pt.z);
    double sg_dist = (goal_pos - start_pos).norm();

    // 采样边界框 (比 A* 大, RRT* 需要更多探索空间)
    const double pad = 100.0;
    double bx_min = std::max(std::min(start_pos.x(), goal_pos.x()) - pad, sdf_x_min);
    double bx_max = std::min(std::max(start_pos.x(), goal_pos.x()) + pad, sdf_x_max);
    double by_min = std::max(std::min(start_pos.y(), goal_pos.y()) - pad, sdf_y_min);
    double by_max = std::min(std::max(start_pos.y(), goal_pos.y()) + pad, sdf_y_max);
    double bz_min = std::max(std::min(start_pos.z(), goal_pos.z()) - pad, sdf_z_min);
    double bz_max = std::min(std::max(start_pos.z(), goal_pos.z()) + pad, sdf_z_max);

    // RRT* 参数
    const double step_size   = std::clamp(sg_dist / 50.0, 10.0, 30.0);  // 扩展步长
    const double goal_bias   = 0.10;     // 10% 概率直接采样目标
    const double goal_thresh = step_size * 1.5;
    const double clearance   = std::min(safety_margin * 0.5, 5.0);
    const double neighbor_r  = step_size * 2.5;  // rewiring 邻居半径
    int rrt_iters = std::max(max_iters * 30, 3000);  // RRT* 需要较多迭代

    std::cout << "[RRT*] 起终点距离 " << sg_dist << " m, 步长 " << step_size
              << " m, 迭代 " << rrt_iters << " 次, 净空 " << clearance << " m\n";

    // 检查起点/终点
    if (query_esdf(start_pos.x(), start_pos.y(), start_pos.z()) < clearance) {
        std::cerr << "[RRT*] 错误: start 在障碍物内或越界!\n";
        return {};
    }
    if (query_esdf(goal_pos.x(), goal_pos.y(), goal_pos.z()) < clearance) {
        std::cerr << "[RRT*] 错误: goal 在障碍物内或越界!\n";
        return {};
    }

    // ==================================================================
    // 2. RRT* 搜索
    // ==================================================================

    // 段碰撞检查: 在两点间插值, 逐点查 ESDF (硬闸门, 保证不穿墙)
    auto is_collision_free = [&](const Vec3& a, const Vec3& b) -> bool {
        double dist = (b - a).norm();
        int steps = std::max(1, (int)std::ceil(dist / (clearance * 0.5)));
        for (int s = 0; s <= steps; ++s) {
            double t = (double)s / steps;
            Vec3 p = a + t * (b - a);
            if (query_esdf(p.x(), p.y(), p.z()) < clearance) return false;
        }
        return true;
    };

    // 风险感知边代价: 段长 × ESDF 风险权重
    // ESDF 风险权重在安全区为 1，贴近障碍时加价。
    // 风场代价系数与 A* 使用同一公式。
    auto seg_cost_with_risk = [&](const Vec3& a, const Vec3& b) -> double {
        double seg_len = (b - a).norm();
        Vec3 mid(0.5 * (a.x() + b.x()),
                 0.5 * (a.y() + b.y()),
                 0.5 * (a.z() + b.z()));
        double d_mid = query_esdf(mid.x(), mid.y(), mid.z());
        double risk_weight = std::max(safety_margin / std::max(d_mid, 0.5), 1.0);
        double cost = seg_len * risk_weight;
        if (wind) {
            cost *= wind->edgeCostCoeff(a.x(), a.y(), a.z(),
                                        b.x(), b.y(), b.z(), dem);
        }
        return cost;
    };

    // 随机数生成器
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(bx_min, bx_max);
    std::uniform_real_distribution<double> uy(by_min, by_max);
    std::uniform_real_distribution<double> uz(bz_min, bz_max);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    // 初始化树
    std::vector<RRTNode> nodes;
    nodes.reserve(rrt_iters + 10);
    nodes.push_back({start_pos, -1, 0.0});

    int goal_node = -1;
    double goal_cost = std::numeric_limits<double>::max();

    for (int iter = 0; iter < rrt_iters; ++iter) {
        if (timed_out()) {
            log_timeout("RRT* 搜索");
            return {};
        }
        // --- 随机采样 (goal-biased) ---
        Vec3 sample;
        if (u01(rng) < goal_bias) {
            sample = goal_pos;
        } else {
            sample = Vec3(ux(rng), uy(rng), uz(rng));
        }

        // --- 找最近邻 ---
        int nearest = 0;
        double nearest_d = (nodes[0].pos - sample).norm();
        for (size_t i = 1; i < nodes.size(); ++i) {
            double d = (nodes[i].pos - sample).norm();
            if (d < nearest_d) { nearest_d = d; nearest = (int)i; }
        }

        // --- 扩展: 从最近邻向采样点方向走 step_size ---
        if (nearest_d < 1e-9) continue;
        Vec3 dir = (sample - nodes[nearest].pos).normalized();
        Vec3 new_pos = nodes[nearest].pos + dir * step_size;

        // 边界检查
        if (new_pos.x() < bx_min || new_pos.x() > bx_max ||
            new_pos.y() < by_min || new_pos.y() > by_max ||
            new_pos.z() < bz_min || new_pos.z() > bz_max) continue;

        // 碰撞检查 (最近邻 → 新节点)
        if (!is_collision_free(nodes[nearest].pos, new_pos)) continue;

        // --- 找邻居 ---
        std::vector<int> neighbors;
        for (size_t i = 0; i < nodes.size(); ++i) {
            if ((nodes[i].pos - new_pos).norm() < neighbor_r)
                neighbors.push_back((int)i);
        }

        // --- 选择最优父节点 (Choose Parent): 边代价纳入风险项 ---
        int best_parent = nearest;
        double best_cost = nodes[nearest].cost + seg_cost_with_risk(nodes[nearest].pos, new_pos);
        for (int idx : neighbors) {
            if (!is_collision_free(nodes[idx].pos, new_pos)) continue;
            double c = nodes[idx].cost + seg_cost_with_risk(nodes[idx].pos, new_pos);
            if (c < best_cost) { best_cost = c; best_parent = idx; }
        }

        // --- 添加新节点 ---
        int new_idx = (int)nodes.size();
        nodes.push_back({new_pos, best_parent, best_cost});

        // --- Rewiring: 尝试通过新节点缩短邻居的代价 (风险感知) ---
        for (int idx : neighbors) {
            if (idx == best_parent) continue;
            double c = best_cost + seg_cost_with_risk(new_pos, nodes[idx].pos);
            if (c < nodes[idx].cost && is_collision_free(new_pos, nodes[idx].pos)) {
                nodes[idx].parent = new_idx;
                nodes[idx].cost = c;
            }
        }

        // --- 检查是否到达目标 (goal 段也纳入风险代价) ---
        if ((new_pos - goal_pos).norm() < goal_thresh) {
            double c = best_cost + seg_cost_with_risk(new_pos, goal_pos);
            if (c < goal_cost) {
                goal_cost = c;
                goal_node = new_idx;
            }
        }

        if (iter % 1000 == 0) {
            std::cout << "[RRT*] iter " << iter << "/" << rrt_iters
                      << ", nodes=" << nodes.size()
                      << (goal_node >= 0 ? " (已到达目标)" : "") << "\n";
        }
    }

    // ==================================================================
    // 3. 路径提取
    // ==================================================================
    std::vector<Vec3> rrt_path;
    if (goal_node >= 0) {
        rrt_path.push_back(goal_pos);
        int idx = goal_node;
        while (idx >= 0) {
            rrt_path.push_back(nodes[idx].pos);
            idx = nodes[idx].parent;
        }
        std::reverse(rrt_path.begin(), rrt_path.end());
        std::cout << "[RRT*] 到达目标, 路径 " << rrt_path.size()
                  << " 个节点 (树节点 " << nodes.size() << ", 代价 " << goal_cost << ")\n";
    } else {
        // 取离 goal 最近的节点
        int best = 0;
        double best_d = (nodes[0].pos - goal_pos).norm();
        for (size_t i = 1; i < nodes.size(); ++i) {
            double d = (nodes[i].pos - goal_pos).norm();
            if (d < best_d) { best_d = d; best = (int)i; }
        }
        if (best_d > 100.0) {
            std::cerr << "[RRT*] 错误: 未收敛, 最近点距 goal " << best_d << " m\n";
            return {};
        }
        int idx = best;
        while (idx >= 0) {
            rrt_path.push_back(nodes[idx].pos);
            idx = nodes[idx].parent;
        }
        std::reverse(rrt_path.begin(), rrt_path.end());
        // 追加连接段
        Vec3 last = rrt_path.back();
        int n_conn = std::max(2, (int)std::ceil(best_d / step_size));
        for (int s = 1; s <= n_conn; ++s) {
            double t = (double)s / n_conn;
            rrt_path.push_back(last + t * (goal_pos - last));
        }
        std::cout << "[RRT*] 近似到达 (距 goal " << best_d << " m), 路径 "
                  << rrt_path.size() << " 节点\n";
    }

    // 确保首尾精确
    rrt_path.front() = start_pos;
    rrt_path.back()  = goal_pos;

    // ==================================================================
    // 4. 后处理: 迭代式 ESDF 投影 + 段间加密 + 3 点平滑
    //    修复: 原单步大跳 (push=margin-d+0.5, d=-4时跳14.5m) 可能落入另一障碍.
    //          改为 FMM 同款迭代式投影 (≤60步, 梯度退化时 +z 兜底) +
    //          段内最差点加密 (避免 RRT* 稀疏路径漏检穿墙).
    // ==================================================================

    // 4a. 单点迭代式 ESDF 投影 (≤60 步, 推到 margin+1m)
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

    // 4b. 路径点投影 + 段内加密 (低采样步长 4m, 检测每段最差点并插入)
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
        std::cout << "[RRT*] ESDF 净空: 投影 " << total_fixed << " 点, 加密 "
                  << total_inserted << " 点 (最终 " << path.size() << " 点)\n";
        return total_fixed + total_inserted;
    };

    if (enable_internal_postprocess) enforce_clear_and_densify(rrt_path);

    // 旧内部后处理仅供兼容；主实验在 main.cpp 统一处理。
    for (int iter = 0; enable_internal_postprocess && iter < 5; ++iter) {
        std::vector<Vec3> smoothed = rrt_path;
        // 内部点: 5 点高斯加权
        for (size_t i = 2; i + 2 < rrt_path.size(); ++i) {
            smoothed[i] = (rrt_path[i-2] + 4.0*rrt_path[i-1] + 6.0*rrt_path[i]
                        + 4.0*rrt_path[i+1] + rrt_path[i+2]) / 16.0;
        }
        // 边界次邻点: 3 点加权 [1,2,1]/4
        if (rrt_path.size() >= 3) {
            size_t last = rrt_path.size() - 1;
            smoothed[1] = (rrt_path[0] + 2.0*rrt_path[1] + rrt_path[2]) * 0.25;
            smoothed[last-1] = (rrt_path[last-2] + 2.0*rrt_path[last-1] + rrt_path[last]) * 0.25;
        }
        rrt_path = smoothed;
        enforce_clear_and_densify(rrt_path);
    }

    rrt_path.front() = start_pos;
    rrt_path.back()  = goal_pos;

    std::cout << "[RRT*] 后处理完成 (ESDF 投影 + 平滑), 路径 "
              << rrt_path.size() << " 个点\n";

    // ==================================================================
    // 5. 输出: 转换为 Node + 统计
    // ==================================================================
    std::vector<Node> final_path;
    final_path.reserve(rrt_path.size());
    for (const Vec3& p : rrt_path)
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

    std::cout << "[RRT*] ===== 最终路径统计 =====\n";
    std::cout << "  - 路径点数      : " << final_path.size() << "\n";
    std::cout << "  - ESDF 距离区间 : [" << min_d << ", " << max_d << "] m\n";
    std::cout << "  - 3D 路径总长   : " << total_3d << " m\n";
    std::cout << "  - 水平路径总长  : " << total_horiz << " m (直线 "
              << straight_horiz << " m, 延伸 "
              << (total_horiz - straight_horiz) << " m)\n";
    std::cout << "  - 最小安全余量  : " << (min_d >= safety_margin ? "OK" : "WARN") << "\n";

    // 导出搜索树用于可视化
    if (out_data) {
        out_data->node_x.reserve(nodes.size());
        out_data->node_y.reserve(nodes.size());
        out_data->node_z.reserve(nodes.size());
        out_data->parent_idx.reserve(nodes.size());
        out_data->cost.reserve(nodes.size());
        for (const auto& n : nodes) {
            out_data->node_x.push_back(n.pos.x());
            out_data->node_y.push_back(n.pos.y());
            out_data->node_z.push_back(n.pos.z());
            out_data->parent_idx.push_back(n.parent);
            out_data->cost.push_back(n.cost);
        }
    }

    return final_path;
}

#endif // RRT_STAR_OPTIMIZER_HPP
