#pragma once

#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include "dem_data.hpp"
#include "itkImage.h"

struct Node {
    double x, y, z;
    int id;
    double cost = 0.0;
    int parent_id = -1;
    double gamma = 0.0; // [新增] 为 Advanced IRRT* 支持爬升角计算[cite: 7]
    
    Node(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

// =========================================================================
// 2. 地理数据查询辅助函数
// =========================================================================

inline double get_distance(const Node& n1, const Node& n2) {
    return std::sqrt(std::pow(n1.x-n2.x, 2) + std::pow(n1.y-n2.y, 2) + std::pow(n1.z-n2.z, 2));
}

using CommonSDFImageType = itk::Image<float, 3>;

enum class PostprocessMode { Raw, Smoothed };

inline std::vector<Node> postprocess_node_path(
    const std::vector<Node>& raw_path, CommonSDFImageType::Pointer sdf_image,
    const DEMData& dem, double safety_margin, PostprocessMode mode,
    double* correction_m = nullptr) {
    if (correction_m) *correction_m = 0.0;
    if (raw_path.size() < 2) return raw_path;

    auto query_esdf = [&](const Node& p) {
        CommonSDFImageType::PointType point;
        point[0] = p.x; point[1] = p.y; point[2] = p.z;
        CommonSDFImageType::IndexType index;
        if (sdf_image->TransformPhysicalPointToIndex(point, index) &&
            sdf_image->GetBufferedRegion().IsInside(index))
            return static_cast<double>(sdf_image->GetPixel(index));
        return -100.0;
    };
    const auto region = sdf_image->GetLargestPossibleRegion();
    const auto size = region.GetSize();
    const auto origin = sdf_image->GetOrigin();
    const auto spacing = sdf_image->GetSpacing();
    const double min_x = origin[0], min_y = origin[1], min_z = origin[2];
    const double max_x = origin[0] + size[0] * spacing[0];
    const double max_y = origin[1] + size[1] * spacing[1];
    const double max_z = origin[2] + size[2] * spacing[2];
    const double gradient_step = std::max({spacing[0], spacing[1], spacing[2]}) * 0.5;

    auto project = [&](Node& p) {
        for (int iteration = 0; iteration < 60; ++iteration) {
            const double distance = query_esdf(p);
            if (distance >= safety_margin) break;
            Node px1(p.x + gradient_step, p.y, p.z), px0(p.x - gradient_step, p.y, p.z);
            Node py1(p.x, p.y + gradient_step, p.z), py0(p.x, p.y - gradient_step, p.z);
            Node pz1(p.x, p.y, p.z + gradient_step), pz0(p.x, p.y, p.z - gradient_step);
            double gx = query_esdf(px1) - query_esdf(px0);
            double gy = query_esdf(py1) - query_esdf(py0);
            double gz = query_esdf(pz1) - query_esdf(pz0);
            double norm = std::sqrt(gx * gx + gy * gy + gz * gz);
            if (norm < 1e-9) { gx = 0.0; gy = 0.0; gz = 1.0; norm = 1.0; }
            const double push = safety_margin - distance + 1.0;
            p.x = std::clamp(p.x + push * gx / norm, min_x, max_x);
            p.y = std::clamp(p.y + push * gy / norm, min_y, max_y);
            p.z = std::clamp(p.z + push * gz / norm, min_z, max_z);
        }
    };

    // 两种模式共享同一安全加密与投影；仅 smoothed 额外执行高斯平滑。
    std::vector<Node> path;
    path.reserve(raw_path.size());
    path.push_back(raw_path.front());
    constexpr double sample_step = 4.0;
    constexpr size_t max_points = 10000;
    for (size_t i = 1; i < raw_path.size() && path.size() < max_points; ++i) {
        const Node& a = raw_path[i - 1];
        const Node& b = raw_path[i];
        const int segments = std::max(
            1, static_cast<int>(std::ceil(get_distance(a, b) / sample_step)));
        for (int k = 1; k <= segments && path.size() < max_points; ++k) {
            const double t = static_cast<double>(k) / segments;
            path.emplace_back(a.x + t * (b.x - a.x),
                              a.y + t * (b.y - a.y),
                              a.z + t * (b.z - a.z));
        }
    }
    const std::vector<Node> before = path;
    for (auto& point : path) project(point);
    const int smoothing_rounds = mode == PostprocessMode::Smoothed ? 5 : 0;
    for (int outer = 0; outer < smoothing_rounds; ++outer) {
        std::vector<Node> source = path;
        for (size_t i = 2; i + 2 < source.size(); ++i) {
            path[i].x = (source[i-2].x + 4*source[i-1].x + 6*source[i].x + 4*source[i+1].x + source[i+2].x) / 16.0;
            path[i].y = (source[i-2].y + 4*source[i-1].y + 6*source[i].y + 4*source[i+1].y + source[i+2].y) / 16.0;
            path[i].z = (source[i-2].z + 4*source[i-1].z + 6*source[i].z + 4*source[i+1].z + source[i+2].z) / 16.0;
        }
        if (source.size() >= 3) {
            const size_t last = source.size() - 1;
            path[1] = Node((source[0].x + 2*source[1].x + source[2].x) * 0.25,
                           (source[0].y + 2*source[1].y + source[2].y) * 0.25,
                           (source[0].z + 2*source[1].z + source[2].z) * 0.25);
            path[last-1] = Node((source[last-2].x + 2*source[last-1].x + source[last].x) * 0.25,
                                (source[last-2].y + 2*source[last-1].y + source[last].y) * 0.25,
                                (source[last-2].z + 2*source[last-1].z + source[last].z) * 0.25);
        }
        for (auto& point : path) project(point);
    }
    if (correction_m) {
        const size_t count = std::min(path.size(), before.size());
        for (size_t i = 0; i < count; ++i) *correction_m += get_distance(path[i], before[i]);
        *correction_m /= static_cast<double>(count);
    }
    return path;
}