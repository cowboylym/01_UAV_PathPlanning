//rrt.hpp
#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <memory>

// 引入 OGR 用于写出 Shp 文件
#include <ogrsf_frmts.h>

// ITK 相关头文件
#include "itkImage.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkImageFileWriter.h"

#include "common.h"

// 定义 ITK 图像类型
using Mask3DImageType = itk::Image<unsigned char, 3>;
using SDFImageType = itk::Image<float, 3>;

// 假设已经获取到 ITK 的 SDF 图像
// using SDFImageType = itk::Image<float, 3>;
// SDFImageType::Pointer sdfImage = distanceFilter->GetOutput();



// // 计算两点间欧氏距离
// double get_distance(const Node& n1, const Node& n2) {
//     return std::sqrt((n1.x - n2.x)*(n1.x - n2.x) + (n1.y - n2.y)*(n1.y - n2.y) + (n1.z - n2.z)*(n1.z - n2.z));
// }

// -------------------------------------------------------------------------
// 2. 基于 ITK 3D SDF 的环境碰撞检测
// -------------------------------------------------------------------------
bool is_state_valid(double x, double y, double z, SDFImageType::Pointer sdfImage, double safe_radius = 5.0) {
    SDFImageType::PointType pt;
    pt[0] = x; pt[1] = y; pt[2] = z;

    SDFImageType::IndexType idx;
    // ITK 自动处理物理坐标到栅格索引的转换（考虑了 Origin 和 Spacing）
    if (!sdfImage->TransformPhysicalPointToIndex(pt, idx)) {
        return false; // 超出 3D 体素场边界，视为不安全
    }

    // 查询有向距离值
    float sdf_val = sdfImage->GetPixel(idx);

    // 如果值小于无人机安全半径（比如5米），说明太靠近山体或在山体内部
    return (sdf_val >= safe_radius);
}

// 检测两点连线是否安全（通过步长离散采样检测）
bool is_segment_valid(const Node& n1, const Node& n2, SDFImageType::Pointer sdfImage, double step_size, double safe_radius) {
    double dist = get_distance(n1, n2);
    int steps = std::ceil(dist / step_size);
    for (int i = 1; i <= steps; ++i) {
        double t = (double)i / steps;
        double cx = n1.x + t * (n2.x - n1.x);
        double cy = n1.y + t * (n2.y - n1.y);
        double cz = n1.z + t * (n2.z - n1.z);
        if (!is_state_valid(cx, cy, cz, sdfImage, safe_radius)) {
            return false;
        }
    }
    return true;
}

// -------------------------------------------------------------------------
// 3. Informed RRT* 算法核心
// -------------------------------------------------------------------------
std::vector<Node> plan_informed_rrt_star(
    Node start, Node goal, SDFImageType::Pointer sdfImage, 
    int max_iter = 5000, double step_len = 15.0, double search_radius = 30.0, double uav_safe_dist = 5.0) 
{
    std::vector<Node> tree;
    tree.push_back(start);

    // 获取 3D 边界以供初始采样（从 ITK 图像物理范围获取）
    SDFImageType::RegionType region = sdfImage->GetLargestPossibleRegion();
    SDFImageType::SizeType size = region.GetSize();
    SDFImageType::SpacingType spacing = sdfImage->GetSpacing();
    SDFImageType::PointType origin = sdfImage->GetOrigin();

    double min_x = origin[0], max_x = origin[0] + size[0] * spacing[0];
    double min_y = origin[1], max_y = origin[1] + size[1] * spacing[1]; // 注意通常地理Y可能朝下，此处简化为包围盒
    double min_z = origin[2], max_z = origin[2] + size[2] * spacing[2];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> rand_x(std::min(min_x, max_x), std::max(min_x, max_x));
    std::uniform_real_distribution<double> rand_y(std::min(min_y, max_y), std::max(min_y, max_y));
    std::uniform_real_distribution<double> rand_z(min_z, max_z);
    std::uniform_real_distribution<double> rand_zero_to_one(0.0, 1.0);

    double c_best = std::numeric_limits<double>::max(); // 当前最优路径长度
    int best_goal_node_id = -1;
    double c_min = get_distance(start, goal);           // 起终点直线距离

    // Informed 采样相关变换矩阵准备
    Node x_center((start.x + goal.x)/2.0, (start.y + goal.y)/2.0, (start.z + goal.z)/2.0);
    // 计算旋转矩阵（将标准椭球对齐到起终点方向）
    // 此处为了代码简洁，直接采用三维空间的基底旋转
    double dx = (goal.x - start.x) / c_min;
    double dy = (goal.y - start.y) / c_min;
    double dz = (goal.z - start.z) / c_min;
    
    // 构造旋转矩阵 R (第一列为起终点方向轴)
    // 简易 Rodrigues 旋转或方向余弦，此处常规 RRT 在超出边界时采样
    // 实际工业应用常在 c_best < inf 后开启椭球内统一随机采样

    for (int iter = 0; iter < max_iter; ++iter) {
        double rx, ry, rz;

        // Informed Sampling 开关
        if (c_best < std::numeric_limits<double>::max()) {
            // 已找到初始解，开启椭球采样
            // 1. 在单位球内均匀采样
            double r = std::cbrt(rand_zero_to_one(gen)); // 径向分布
            double theta = rand_zero_to_one(gen) * M_PI;
            double phi = rand_zero_to_one(gen) * 2.0 * M_PI;
            double x_ball = r * std::sin(theta) * std::cos(phi);
            double y_ball = r * std::sin(theta) * std::sin(phi);
            double z_ball = r * std::cos(theta);

            // 2. 拉伸为椭球
            double r1 = c_best / 2.0;
            double r2 = std::sqrt(c_best*c_best - c_min*c_min) / 2.0;
            double r3 = r2;
            double x_ellipse = r1 * x_ball;
            double y_ellipse = r2 * y_ball;
            double z_ellipse = r3 * z_ball;

            // 3. 旋转并平移回世界坐标系（简化版本，直接沿大方向向量累加，严谨做法需乘完整的张量旋转矩阵）
            // 这里用标准近似：直接检测如果不在采样盒内则退化，或直接使用坐标系变换：
            // 为保证代码严密和不引入庞大的矩阵库，此处混入全局采样作为补偿：
            if(rand_zero_to_one(gen) > 0.2) {
                // 严谨的 3D 旋转变换通常需要仿射变换，此处若简化可在全局采样里通过距离剪枝：
                rx = rand_x(gen); ry = rand_y(gen); rz = rand_z(gen);
                if (std::sqrt((rx-start.x)*(rx-start.x)+(ry-start.y)*(ry-start.y)+(rz-start.z)*(rz-start.z)) + 
                    std::sqrt((rx-goal.x)*(rx-goal.x)+(ry-goal.y)*(ry-goal.y)+(rz-goal.z)*(rz-goal.z)) > c_best) {
                    continue; // 剪枝过滤不符合椭球要求的点
                }
            } else {
                rx = rand_x(gen); ry = rand_y(gen); rz = rand_z(gen);
            }
        } else {
            // 未找到初始路径前，全空间均匀采样
            rx = rand_x(gen); ry = rand_y(gen); rz = rand_z(gen);
        }

        // 目标偏置（5% 概率直接采样终点加速收敛）
        if (rand_zero_to_one(gen) < 0.05) {
            rx = goal.x; ry = goal.y; rz = goal.z;
        }

        Node q_rand(rx, ry, rz);

        // 找到树上最近的节点
        int nearest_id = 0;
        double min_d = get_distance(tree[0], q_rand);
        for (size_t i = 1; i < tree.size(); ++i) {
            double d = get_distance(tree[i], q_rand);
            if (d < min_d) {
                min_d = d;
                nearest_id = i;
            }
        }
        Node q_nearest = tree[nearest_id];

        // 步进控制（Steer）
        if (min_d > step_len) {
            double theta_x = (q_rand.x - q_nearest.x) / min_d;
            double theta_y = (q_rand.y - q_nearest.y) / min_d;
            double theta_z = (q_rand.z - q_nearest.z) / min_d;
            q_rand.x = q_nearest.x + step_len * theta_x;
            q_rand.y = q_nearest.y + step_len * theta_y;
            q_rand.z = q_nearest.z + step_len * theta_z;
        }

        // 碰撞检测（利用 SDF 极速查询）
        if (!is_segment_valid(q_nearest, q_rand, sdfImage, 1.0, uav_safe_dist)) {
            continue;
        }

        // RRT* 核心 1：寻找范围内的近邻，并选择代价最小的父亲
        int parent_id = nearest_id;
        double min_cost = q_nearest.cost + get_distance(q_nearest, q_rand);
        std::vector<int> neighbors;

        for (size_t i = 0; i < tree.size(); ++i) {
            double d = get_distance(tree[i], q_rand);
            if (d < search_radius) {
                neighbors.push_back(i);
                double c = tree[i].cost + d;
                if (c < min_cost && is_segment_valid(tree[i], q_rand, sdfImage, 2.0, uav_safe_dist)) {
                    min_cost = c;
                    parent_id = i;
                }
            }
        }

        q_rand.cost = min_cost;
        q_rand.parent_id = parent_id;
        tree.push_back(q_rand);
        int new_node_id = tree.size() - 1;

        // RRT* 核心 2：重构树（Rewire）
        for (int n_id : neighbors) {
            double d = get_distance(tree[new_node_id], tree[n_id]);
            if (tree[new_node_id].cost + d < tree[n_id].cost) {
                if (is_segment_valid(tree[new_node_id], tree[n_id], sdfImage, 2.0, uav_safe_dist)) {
                    tree[n_id].parent_id = new_node_id;
                    tree[n_id].cost = tree[new_node_id].cost + d;
                }
            }
        }

        // 检查是否可以连接到终点
        double dist_to_goal = get_distance(tree[new_node_id], goal);
        if (dist_to_goal < step_len) {
            if (is_segment_valid(tree[new_node_id], goal, sdfImage, 2.0, uav_safe_dist)) {
                double total_c = tree[new_node_id].cost + dist_to_goal;
                if (total_c < c_best) {
                    c_best = total_c;
                    best_goal_node_id = new_node_id;
                    std::cout << "[Informed RRT*] 迭代 " << iter << " 发现更优路径，总长度: " << c_best << " 米\n";
                }
            }
        }
    }

    // 回溯生成最终路径
    std::vector<Node> path;
    if (best_goal_node_id == -1) {
        std::cerr << "规划失败：在给定迭代次数内未找到可行安全航线！\n";
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

// -------------------------------------------------------------------------
// 4. 将路径点导出为含有 Z 值的 3D Shapefile (LineString25D)
// -------------------------------------------------------------------------
void save_path_to_shapefile(const std::vector<Node>& path, const std::string& shp_path) {
    GDALAllRegister();
    GDALDriver* poDriver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
    if (!poDriver) {
        std::cerr << "错误：无法加载 ESRI Shapefile 驱动！" << std::endl;
        return;
    }

    GDALDataset* poDS = poDriver->Create(shp_path.c_str(), 0, 0, 0, GDT_Unknown, nullptr);
    if (!poDS) {
        std::cerr << "错误：创建 Shp 文件失败！" << std::endl;
        return;
    }

    OGRLayer* poLayer = poDS->CreateLayer("uav_path", nullptr, wkbLineString25D, nullptr);
    if (!poLayer) {
        std::cerr << "错误：创建图层失败！" << std::endl;
        GDALClose(poDS);
        return;
    }

    // 给 Shapefile 属性表添加一个字段用来存节点序号
    OGRFieldDefn oField("Node_ID", OFTInteger);
    if (poLayer->CreateField(&oField) != OGRERR_NONE) {
        std::cerr << "警告：创建属性字段失败！" << std::endl;
    }

    // 创建一条 3D 折线几何体
    OGRLineString* poLine = new OGRLineString();
    for (const auto& node : path) {
        poLine->addPoint(node.x, node.y, node.z); // 写入 SPCS 投影坐标 X, Y 和高程 Z
    }

    // 将几何体打包进 Feature
    OGRFeature* poFeature = OGRFeature::CreateFeature(poLayer->GetLayerDefn());
    poFeature->SetGeometryDirectly(poLine); // 转移所有权给 Feature
    poFeature->SetField("Node_ID", 1);

    if (poLayer->CreateFeature(poFeature) != OGRERR_NONE) {
        std::cerr << "错误：向 Shapefile 写入要素失败！" << std::endl;
    }

    // 释放资源，持久化文件到磁盘
    OGRFeature::DestroyFeature(poFeature);
    GDALClose(poDS);
    std::cout << ">>> 航线成功导出至 3D Shapefile 文件: " << shp_path << std::endl;
}

