#ifndef ADVANCED_IRRT_HPP
#define ADVANCED_IRRT_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <memory>

// OGR 用于写出 Shp 文件
#include <ogrsf_frmts.h>

// ITK 相关头文件
#include "itkImage.h"
#include "itkImageFileReader.h"

// 包含全局数据类型
#include "common.h"
#include "dem_data.hpp"

// =========================================================================
// 1. 类型定义与基础结构
// =========================================================================

// 定义 ITK 图像类型 (3D 为 ESDF)
using SDFImageType = itk::Image<float, 3>;

// 算法权重与参数配置
struct IRRTConfig {
    // 边代价权重
    double w1 = 1.0;  // 长度
    double w2 = 30.0; // 爬升
    double w3 = 15.0; // 地形跟随
    double w4 = 20.0; // 平滑度
    double w5 = 10.0; // 障碍物接近

    // 地形跟随参数
    double h0 = 15.0; // 基础离地高度
    double k_slope = 30.0; // 地形敏感系数
    
    // 爬升惩罚
    double lambda_down = 1.5; // 下降惩罚系数
    double gamma_max = 30.0 * M_PI / 180.0; // 最大爬升角

    // 障碍物惩罚衰减
    double sigma_obs = 3.0;
    double d_safe = 10.0; // ESDF 绝对安全距离

    // 采样参数
    double step_min = 5.0;
    double step_max = 100.0; // 建议最大跨度设为较大值
};

// 查询 3D ESDF 距离
float query_esdf(SDFImageType::Pointer sdf, double x, double y, double z) {
    SDFImageType::PointType pt; pt[0]=x; pt[1]=y; pt[2]=z;
    SDFImageType::IndexType idx;
    if (sdf->TransformPhysicalPointToIndex(pt, idx)) return sdf->GetPixel(idx);
    return -1.0; // 越界视为碰撞
}

// 查询 2D 数据
float query_2d(const DEMData& img, double x, double y) {
    return img.getElevation(x, y); 
}

// 计算自适应安全高度 h_safe
double get_safe_height(double x, double y, const DEMData& dem, const DEMData& slope, const IRRTConfig& cfg) {
    double H = query_2d(dem, x, y);
    double angle_deg = query_2d(slope, x, y); 
    double angle_rad = angle_deg * M_PI / 180.0;
    double tan_S = std::tan(angle_rad); 
    return H + cfg.h0 + cfg.k_slope * tan_S;
}

// =========================================================================
// 2. 代价计算与硬约束模块
// =========================================================================

// MCMC 采样状态代价 (点代价)
double evaluate_state_cost(double x, double y, double z, SDFImageType::Pointer sdf, 
                           const DEMData& dem, const DEMData& slope, const IRRTConfig& cfg) {
    double d = query_esdf(sdf, x, y, z);
    if (d < cfg.d_safe) return 1e6; // 碰撞点代价无限大

    double h_safe = get_safe_height(x, y, dem, slope, cfg);
    double P = std::abs(z - h_safe);
    
    // 归一化 (假定最大偏离 100m，最大距离 20m)
    double P_norm = std::min(P / 100.0, 1.0);
    double d_norm = std::min(d / 20.0, 1.0);
    
    return 10.0 * (P_norm * P_norm) - 5.0 * d_norm;
}

// 边代价函数 (5项多目标代价)
double calculate_edge_cost(const Node& n_parent, const Node& n_child, 
                           SDFImageType::Pointer sdf, const DEMData& dem, const DEMData& slope, 
                           const IRRTConfig& cfg) {
    double L = get_distance(n_parent, n_child);
    if (L < 1e-3) return 0.0;

    // 1. 爬升代价 C_gamma
    double delta_z = n_child.z - n_parent.z;
    double C_up = delta_z > 0 ? (delta_z * delta_z) : 0;
    double C_down = delta_z < 0 ? (cfg.lambda_down * delta_z * delta_z) : 0;
    double C_gamma = C_up + C_down;

    // 2. 地形跟随代价 C_P
    double h_safe = get_safe_height(n_child.x, n_child.y, dem, slope, cfg);
    double C_P = std::pow(n_child.z - h_safe, 2);

    // 3. 平滑代价 C_delta_gamma
    double gamma_new = std::asin(std::clamp(delta_z / L, -1.0, 1.0));
    double C_smooth = std::pow(gamma_new - n_parent.gamma, 2);

    // 4. 障碍物接近代价 C_obs (使用中点近似)
    double mid_x = (n_parent.x + n_child.x) / 2.0;
    double mid_y = (n_parent.y + n_child.y) / 2.0;
    double mid_z = (n_parent.z + n_child.z) / 2.0;
    double d_mid = query_esdf(sdf, mid_x, mid_y, mid_z);
    double C_obs = std::exp(-std::max(0.0, d_mid) / cfg.sigma_obs);

    return cfg.w1 * L + cfg.w2 * C_gamma + cfg.w3 * C_P + cfg.w4 * C_smooth + cfg.w5 * C_obs;
}

// 硬约束检查：碰撞检测 + 最大爬升角
bool hard_constraint_check(const Node& n1, const Node& n2, SDFImageType::Pointer sdf, const IRRTConfig& cfg) {
    double L = get_distance(n1, n2);
    if (L < 1e-3) return false;

    // 爬升角硬约束
    double gamma = std::asin(std::clamp(std::abs(n2.z - n1.z) / L, 0.0, 1.0));
    if (gamma > cfg.gamma_max) return false;

    // 离散碰撞检测
    int steps = std::ceil(L / 1.0); // 1m 步长
    for (int i = 1; i <= steps; ++i) {
        double t = (double)i / steps;
        double cx = n1.x + t * (n2.x - n1.x);
        double cy = n1.y + t * (n2.y - n1.y);
        double cz = n1.z + t * (n2.z - n1.z);
        if (query_esdf(sdf, cx, cy, cz) < cfg.d_safe) return false;
    }
    return true;
}

// =========================================================================
// 3. ADVANCED_IRRT* 主函数
// =========================================================================

std::vector<Node> plan_advanced_irrt_star(
    Node start, Node goal, 
    SDFImageType::Pointer sdf, const DEMData& dem, const DEMData& slope,
    int max_iter = 5000, 
    double step_size = 100.0, 
    double search_radius = 150.0, // 增加搜索半径，确保局部重连范围能盖过最大步长
    double safety_margin = 10.0)
{
    IRRTConfig cfg;
    cfg.step_max = step_size; 
    cfg.step_min = std::max(10.0, step_size / 3.0); 
    cfg.d_safe = safety_margin; 

    std::vector<Node> tree;
    tree.push_back(start);

    auto region = sdf->GetLargestPossibleRegion();
    auto size = region.GetSize();
    auto spacing = sdf->GetSpacing();
    auto origin = sdf->GetOrigin();
    double min_x = origin[0], max_x = origin[0] + size[0] * spacing[0];
    double min_y = origin[1], max_y = origin[1] + size[1] * spacing[1];
    double min_z = origin[2], max_z = origin[2] + size[2] * spacing[2];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> rand_x(min_x, max_x);
    std::uniform_real_distribution<double> rand_y(min_y, max_y);
    std::uniform_real_distribution<double> rand_z(min_z, max_z);
    std::uniform_real_distribution<double> rand_01(0.0, 1.0);

    double c_best = std::numeric_limits<double>::max();
    int best_goal_node_id = -1;
    
    // MCMC 变量
    double T_mcmc = 2.0; 
    double lambda_T = 0.005; 
    Node mcmc_state = start;
    double current_state_cost = evaluate_state_cost(mcmc_state.x, mcmc_state.y, mcmc_state.z, sdf, dem, slope, cfg);

    for (int iter = 0; iter < max_iter; ++iter) {
        
        Node q_rand(0, 0, 0);
        double p = rand_01(gen);

        // 策略分发：平衡探索与利用
        if (p < 0.10) {
            // [10%] 目标偏置
            q_rand = goal;
        } 
        else if (p < 0.40) {
            // [30%] 全局随机（基于安全高度约束 Z，防止爬升角频繁触发）
            q_rand.x = rand_x(gen);
            q_rand.y = rand_y(gen);
            double safe_z = get_safe_height(q_rand.x, q_rand.y, dem, slope, cfg);
            std::normal_distribution<double> z_noise(0.0, 15.0); 
            q_rand.z = std::clamp(safe_z + z_noise(gen), min_z, max_z);
        } 
        else {
            // [60%] 启发式/MCMC
            double current_slope_deg = query_2d(slope, mcmc_state.x, mcmc_state.y);
            bool is_flat_terrain = (current_slope_deg < 8.0);

            if (is_flat_terrain) {
                // 平坦区：起点到终点的定向宽松通道
                std::uniform_real_distribution<double> rand_ratio(0.0, 1.0);
                double t_ratio = rand_ratio(gen);
                double base_x = start.x + t_ratio * (goal.x - start.x);
                double base_y = start.y + t_ratio * (goal.y - start.y);
                double base_z = start.z + t_ratio * (goal.z - start.z);

                std::normal_distribution<double> line_noise(0.0, 80.0); // 通道给足 80m 横纵余量
                q_rand.x = std::clamp(base_x + line_noise(gen), min_x, max_x);
                q_rand.y = std::clamp(base_y + line_noise(gen), min_y, max_y);
                q_rand.z = std::clamp(base_z + line_noise(gen), min_z, max_z);
            } else {
                // 复杂/陡峭区：MCMC 局部自适应扰动搜索
                double current_esdf_d = query_esdf(sdf, mcmc_state.x, mcmc_state.y, mcmc_state.z);
                if (current_esdf_d < 0) current_esdf_d = cfg.d_safe;
                
                double adaptive_sigma = std::clamp(current_esdf_d, 10.0, 50.0); 
                std::normal_distribution<double> adaptive_gaussian(0.0, adaptive_sigma);

                Node q_prop(mcmc_state.x + adaptive_gaussian(gen), 
                            mcmc_state.y + adaptive_gaussian(gen), 
                            mcmc_state.z + adaptive_gaussian(gen));
                
                q_prop.x = std::clamp(q_prop.x, min_x, max_x);
                q_prop.y = std::clamp(q_prop.y, min_y, max_y);
                q_prop.z = std::clamp(q_prop.z, min_z, max_z);

                double prop_cost = evaluate_state_cost(q_prop.x, q_prop.y, q_prop.z, sdf, dem, slope, cfg);
                double A = std::exp(-(prop_cost - current_state_cost) / T_mcmc);
                if (rand_01(gen) < A) {
                    mcmc_state = q_prop;
                    current_state_cost = prop_cost;
                }
                q_rand = mcmc_state;
                T_mcmc = std::max(0.1, 2.0 * std::exp(-lambda_T * iter));
            }
        }

        // =====================================================================
        // RRT* 扩展过程
        // =====================================================================
        Node q_new = q_rand;

        // Nearest
        int nearest_id = 0;
        double min_d = get_distance(tree[0], q_new);
        for (size_t i = 1; i < tree.size(); ++i) {
            double d = get_distance(tree[i], q_new);
            if (d < min_d) { min_d = d; nearest_id = i; }
        }
        Node q_nearest = tree[nearest_id];

        // Steer
        double esdf_d = query_esdf(sdf, q_nearest.x, q_nearest.y, q_nearest.z);
        double safe_esdf = std::max(0.0, esdf_d);
        double current_slope_deg = query_2d(slope, q_nearest.x, q_nearest.y);
        bool is_flat_terrain = (current_slope_deg < 8.0);
        
        double current_max_step = is_flat_terrain ? cfg.step_max : (cfg.step_max * 0.7);
        double step_len = std::clamp(cfg.step_min + 1.5 * safe_esdf, cfg.step_min, current_max_step);

        if (min_d > step_len) {
            double theta_x = (q_new.x - q_nearest.x) / min_d;
            double theta_y = (q_new.y - q_nearest.y) / min_d;
            double theta_z = (q_new.z - q_nearest.z) / min_d;
            q_new.x = q_nearest.x + step_len * theta_x;
            q_new.y = q_nearest.y + step_len * theta_y;
            q_new.z = q_nearest.z + step_len * theta_z;
        }

        // 碰撞与硬约束检查 (过陡峭则直接抛弃)
        if (!hard_constraint_check(q_nearest, q_new, sdf, cfg)) continue;

        // Choose Parent
        int parent_id = nearest_id;
        double min_cost = q_nearest.cost + calculate_edge_cost(q_nearest, q_new, sdf, dem, slope, cfg);
        std::vector<int> neighbors;

        for (size_t i = 0; i < tree.size(); ++i) {
            double d = get_distance(tree[i], q_new);
            if (d < search_radius) {
                neighbors.push_back(i);
                // 判断备选父节点硬约束
                if (hard_constraint_check(tree[i], q_new, sdf, cfg)) {
                    double c = tree[i].cost + calculate_edge_cost(tree[i], q_new, sdf, dem, slope, cfg);
                    if (c < min_cost) { min_cost = c; parent_id = i; }
                }
            }
        }

        q_new.cost = min_cost;
        q_new.parent_id = parent_id;
        
        // 防御性安全检查，防止除以 0 导致 asin 越界
        double dist_to_parent = get_distance(tree[parent_id], q_new);
        q_new.gamma = dist_to_parent > 1e-3 ? 
            std::asin(std::clamp((q_new.z - tree[parent_id].z) / dist_to_parent, -1.0, 1.0)) : 0.0;
        
        tree.push_back(q_new);
        int new_node_id = tree.size() - 1;

        // Rewire (重连网格优化)
        for (int n_id : neighbors) {
            if (n_id == parent_id) continue;
            if (hard_constraint_check(tree[new_node_id], tree[n_id], sdf, cfg)) {
                double rewire_cost = tree[new_node_id].cost + calculate_edge_cost(tree[new_node_id], tree[n_id], sdf, dem, slope, cfg);
                if (rewire_cost < tree[n_id].cost) {
                    tree[n_id].parent_id = new_node_id;
                    tree[n_id].cost = rewire_cost;
                    
                    double dist_n = get_distance(tree[new_node_id], tree[n_id]);
                    tree[n_id].gamma = dist_n > 1e-3 ? 
                        std::asin(std::clamp((tree[n_id].z - tree[new_node_id].z) / dist_n, -1.0, 1.0)) : 0.0;
                }
            }
        }

        // 检查终点连接：放宽连接阈值，只要在单次最大步长且无碰撞即可直连终点
        double dist_to_goal = get_distance(tree[new_node_id], goal);
        if (dist_to_goal <= cfg.step_max && hard_constraint_check(tree[new_node_id], goal, sdf, cfg)) {
            double total_c = tree[new_node_id].cost + calculate_edge_cost(tree[new_node_id], goal, sdf, dem, slope, cfg);
            if (total_c < c_best) {
                c_best = total_c;
                best_goal_node_id = new_node_id;
            }
        }
    }

    // 回溯路径
    std::vector<Node> path;
    if (best_goal_node_id == -1) {
        std::cerr << "规划失败：在限制迭代次数内未能找到安全且连通的航线！" << std::endl;
        return path;
    }
    
    goal.parent_id = best_goal_node_id;
    path.push_back(goal);
    
    int curr_id = best_goal_node_id;
    while (curr_id != -1) {
        path.push_back(tree[curr_id]);
        curr_id = tree[curr_id].parent_id;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

#endif