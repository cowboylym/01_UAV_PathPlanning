// rrt_dem.hpp
#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <memory>

#include "common.h"
#include "dem_collision.hpp"  // 包含 DEMData 和碰撞检测函数

// 如果 M_PI 未定义，定义之
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 计算两点间欧氏距离（与 rrt.hpp 中相同）
inline double get_distance_dem(const Node& n1, const Node& n2) {
    return std::sqrt((n1.x - n2.x)*(n1.x - n2.x) + 
                     (n1.y - n2.y)*(n1.y - n2.y) + 
                     (n1.z - n2.z)*(n1.z - n2.z));
}

// 基于 DEM 的 Informed RRT* 规划器
inline std::vector<Node> plan_informed_rrt_star_dem(
    Node start, 
    Node goal, 
    const DEMData& dem,
    int max_iter = 5000, 
    double step_len = 15.0, 
    double search_radius = 30.0, 
    double safety_margin = 5.0) 
{
    std::vector<Node> tree;
    tree.push_back(start);

    // 获取 DEM 的物理边界用于随机采样
    double min_x = dem.geoTransform[0];
    double max_x = dem.geoTransform[0] + dem.width * dem.geoTransform[1];
    double min_y = dem.geoTransform[3] + dem.height * dem.geoTransform[5];
    double max_y = dem.geoTransform[3];
    // Z 范围需根据实际地形设定，这里简单从 DEM 最大最小高程 + 缓冲区获取（但 DEMData 未提供最大最小，可自行计算）
    // 为简化，假设用户已知，或从外部传入，这里直接给一个宽范围，但更好的做法是统计 DEM 高程极值
    // 由于我们只有 getElevation，可以遍历所有像素统计，但效率低，此处假定外部传入 min_z, max_z 作为参数，或者我们在这里计算。
    // 为了代码完整性，我们在这里遍历 DEM 统计一次（但 DEMData 没有提供极值，我们可以新增函数，但为了简化，我们可以在外部计算后传入）。
    // 但函数签名固定，为了不破坏接口，我们在这里计算一次，但注意 DEMData 的 data 是 float*，可访问。
    // 我们添加一个辅助函数来获取极值。
    // 这里我们定义一个局部 lambda 来获取 min/max Z
    double min_z = 0.0, max_z = 500.0; // 默认，若 DEM 实际高程超出则可能采样不足，但碰撞检测会过滤。
    // 更好的做法是从 DEM 数据中统计
    bool has_elev = false;
    for (int i = 0; i < dem.height; ++i) {
        for (int j = 0; j < dem.width; ++j) {
            float val = dem.data[i * dem.width + j];
            if (dem.hasNodata && std::abs(val - dem.nodata) < 1e-3) continue;
            if (!has_elev) { min_z = max_z = val; has_elev = true; }
            else { if (val < min_z) min_z = val; if (val > max_z) max_z = val; }
        }
    }
    if (!has_elev) { min_z = 0; max_z = 100; } // fallback
    // 增加一些缓冲区，使得规划高度可以略高于最高点
    double buffer = 50.0;
    min_z = std::floor(min_z) - 10.0; // 略微向下
    max_z = std::ceil(max_z) + buffer;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> rand_x(min_x, max_x);
    std::uniform_real_distribution<double> rand_y(min_y, max_y);
    std::uniform_real_distribution<double> rand_z(min_z, max_z);
    std::uniform_real_distribution<double> rand_zero_to_one(0.0, 1.0);
    std::uniform_real_distribution<double> rand_ball(-1.0, 1.0);

    double c_best = std::numeric_limits<double>::max();
    int best_goal_node_id = -1;
    double c_min = get_distance_dem(start, goal);

    // Informed 采样相关 (旋转矩阵)
    double x_center = (start.x + goal.x) / 2.0;
    double y_center = (start.y + goal.y) / 2.0;
    double z_center = (start.z + goal.z) / 2.0;
    double dx = goal.x - start.x;
    double dy = goal.y - start.y;
    double dz = goal.z - start.z;
    double yaw = std::atan2(dy, dx);
    double pitch = std::atan2(-dz, std::sqrt(dx*dx + dy*dy));

    for (int iter = 0; iter < max_iter; ++iter) {
        double rx, ry, rz;

        // Informed Sampling
        if (c_best < std::numeric_limits<double>::max()) {
            // 在单位球内采样
            double u, v, w;
            do {
                u = rand_ball(gen); v = rand_ball(gen); w = rand_ball(gen);
            } while (u*u + v*v + w*w > 1.0);
            double r1 = c_best / 2.0;
            double r2 = std::sqrt(c_best*c_best - c_min*c_min) / 2.0;
            double r3 = r2;
            double x_ellipse = r1 * u;
            double y_ellipse = r2 * v;
            double z_ellipse = r3 * w;
            // 旋转平移
            rx = x_ellipse * std::cos(pitch) * std::cos(yaw) 
               - y_ellipse * std::sin(yaw) 
               + z_ellipse * std::sin(pitch) * std::cos(yaw) + x_center;
            ry = x_ellipse * std::cos(pitch) * std::sin(yaw) 
               + y_ellipse * std::cos(yaw) 
               + z_ellipse * std::sin(pitch) * std::sin(yaw) + y_center;
            rz = -x_ellipse * std::sin(pitch) + z_ellipse * std::cos(pitch) + z_center;
            // 若超出边界则重新采样（这里简单跳过，继续循环）
            if (rx < min_x || rx > max_x || ry < min_y || ry > max_y || rz < min_z || rz > max_z) {
                continue;
            }
        } else {
            // 全局采样
            rx = rand_x(gen);
            ry = rand_y(gen);
            rz = rand_z(gen);
        }

        // 目标偏置
        if (rand_zero_to_one(gen) < 0.05) {
            rx = goal.x; ry = goal.y; rz = goal.z;
        }

        Node q_rand(rx, ry, rz);

        // 找最近节点
        int nearest_id = 0;
        double min_d = get_distance_dem(tree[0], q_rand);
        for (size_t i = 1; i < tree.size(); ++i) {
            double d = get_distance_dem(tree[i], q_rand);
            if (d < min_d) {
                min_d = d;
                nearest_id = i;
            }
        }
        Node q_nearest = tree[nearest_id];

        // Steer
        if (min_d > step_len) {
            double theta_x = (q_rand.x - q_nearest.x) / min_d;
            double theta_y = (q_rand.y - q_nearest.y) / min_d;
            double theta_z = (q_rand.z - q_nearest.z) / min_d;
            q_rand.x = q_nearest.x + step_len * theta_x;
            q_rand.y = q_nearest.y + step_len * theta_y;
            q_rand.z = q_nearest.z + step_len * theta_z;
        }

        // 碰撞检测（DEM 射线查询）
        if (!is_segment_valid_dem(q_nearest, q_rand, dem, 1.0, safety_margin)) {
            continue;
        }

        // RRT* 选择父节点
        int parent_id = nearest_id;
        double min_cost = q_nearest.cost + get_distance_dem(q_nearest, q_rand);
        std::vector<int> neighbors;
        for (size_t i = 0; i < tree.size(); ++i) {
            double d = get_distance_dem(tree[i], q_rand);
            if (d < search_radius) {
                neighbors.push_back(i);
                double c = tree[i].cost + d;
                if (c < min_cost && is_segment_valid_dem(tree[i], q_rand, dem, 1.0, safety_margin)) {
                    min_cost = c;
                    parent_id = i;
                }
            }
        }

        q_rand.cost = min_cost;
        q_rand.parent_id = parent_id;
        tree.push_back(q_rand);
        int new_node_id = tree.size() - 1;

        // Rewire
        for (int n_id : neighbors) {
            double d = get_distance_dem(tree[new_node_id], tree[n_id]);
            if (tree[new_node_id].cost + d < tree[n_id].cost) {
                if (is_segment_valid_dem(tree[new_node_id], tree[n_id], dem, 1.0, safety_margin)) {
                    tree[n_id].parent_id = new_node_id;
                    tree[n_id].cost = tree[new_node_id].cost + d;
                }
            }
        }

        // 尝试连接到目标
        double dist_to_goal = get_distance_dem(tree[new_node_id], goal);
        if (dist_to_goal < step_len) {
            if (is_segment_valid_dem(tree[new_node_id], goal, dem, 1.0, safety_margin)) {
                double total_c = tree[new_node_id].cost + dist_to_goal;
                if (total_c < c_best) {
                    c_best = total_c;
                    best_goal_node_id = new_node_id;
                    std::cout << "[Informed RRT* DEM] 迭代 " << iter << " 发现更优路径，总长度: " << c_best << " 米\n";
                }
            }
        }
    }

    std::vector<Node> path;
    if (best_goal_node_id == -1) {
        std::cerr << "规划失败：在给定迭代次数内未找到可行安全航线 (DEM)!\n";
        return path;
    }

    path.push_back(goal);
    int curr_id = best_goal_node_id;
    while (curr_id != -1) {
        path.push_back(tree[curr_id]);
        curr_id = tree[curr_id].parent_id;
    }
    std::reverse(path.begin(), path.end());
    return path;
}