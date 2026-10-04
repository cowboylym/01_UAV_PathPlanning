// main.cpp
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "dem_data.hpp"
#include "fmm_optimizer.hpp"                // FMM 规划器
#include "fgdastar_optimizer.hpp"          // FGDA* 规划器
#include "astar_optimizer.hpp"             // A* 规划器
#include "rrt_star_optimizer.hpp"          // RRT* 规划器
#include "common.h"                         // 包含全局 Node 结构体
#include "wind_field.hpp"                   // [新增] 三维风场查询 (节能航线规划)
#include "buildESDF.hpp"
// #include "geojson_elevation.hpp"           // GeoJSON 高程补全工具 (已注释)

#include <gdal_priv.h>
#include <ogrsf_frmts.h>                    // OGR 读取 GeoJSON

#include <chrono>
#include <cstring>   // for stat
#include <numeric>   // for std::iota
#include <sys/stat.h>                        // [新增] mkdir 创建实验输出目录
#include <fstream>   // for CSV 输出
#include <iomanip>   // for setprecision
#include <iostream>
#include <sstream>   // for ostringstream
#include <filesystem>
#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <limits>

// ===== 实验模式控制 =====
// ONLY_SHP_PAIR=1: 仅运行 shp_save 起终点组 (快速, 用于搜索过程可视化)
// ONLY_SHP_PAIR=0: 运行全排列实验 (完整对比分析)
#define ONLY_SHP_PAIR 0

// 定义枚举
enum PlannerType { FGDASTAR_PLANNER = 0, FMM_PLANNER = 1, ASTAR_PLANNER = 2, RRT_STAR_PLANNER = 3 };

// 辅助函数：检查文件是否存在
inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

// 从 GeoTIFF 加载 DEMData（深拷贝数据）
bool loadDEMDataFromFile(const std::string& filename, DEMData& dem) {
    GDALAllRegister();
    GDALDataset* poDS = (GDALDataset*)GDALOpen(filename.c_str(), GA_ReadOnly);
    if (!poDS) return false;

    int width = poDS->GetRasterXSize();
    int height = poDS->GetRasterYSize();
    float* data = new float[width * height];
    poDS->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, data, width, height, GDT_Float32, 0, 0);

    int has_nodata = 0;
    double nodata = poDS->GetRasterBand(1)->GetNoDataValue(&has_nodata);
    double min_z = 1e30, max_z = -1e30;
    bool has_valid = false;
    for (int i = 0; i < width * height; ++i) {
        float val = data[i];
        if (has_nodata && std::abs(val - nodata) < 1e-3) continue;
        if (!has_valid) {
            min_z = max_z = val;
            has_valid = true;
        } else {
            if (val < min_z) min_z = val;
            if (val > max_z) max_z = val;
        }
    }
    if (!has_valid) {
        delete[] data;
        GDALClose(poDS);
        return false;
    }

    dem.width = width;
    dem.height = height;
    dem.data = data;               
    dem.min_elevation = min_z;
    dem.max_elevation = max_z;
    dem.hasNodata = (has_nodata != 0);
    dem.nodata = nodata;
    poDS->GetGeoTransform(dem.geoTransform);

    GDALClose(poDS);
    return true;
}

// 从 GeoJSON 加载 3D 点 (返回 vector<Node>, 失败返回空)
std::vector<Node> load_points_from_geojson(const std::string& geojson_path) {
    std::vector<Node> points;
    GDALAllRegister();
    GDALDataset* poDS = (GDALDataset*)GDALOpenEx(
        geojson_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr);
    if (!poDS) {
        std::cerr << "[GeoJSON] 无法打开: " << geojson_path << "\n";
        return points;
    }
    if (poDS->GetLayerCount() < 1) {
        std::cerr << "[GeoJSON] 无图层\n";
        GDALClose(poDS);
        return points;
    }
    OGRLayer* poLayer = poDS->GetLayer(0);
    poLayer->ResetReading();
    OGRFeature* poFeat;
    while ((poFeat = poLayer->GetNextFeature()) != nullptr) {
        OGRGeometry* poGeom = poFeat->GetGeometryRef();
        if (poGeom && wkbFlatten(poGeom->getGeometryType()) == wkbPoint) {
            OGRPoint* poPt = poGeom->toPoint();
            double x = poPt->getX();
            double y = poPt->getY();
            double z = poPt->Is3D() ? poPt->getZ() : 0.0;
            points.push_back(Node(x, y, z));
        }
        OGRFeature::DestroyFeature(poFeat);
    }
    GDALClose(poDS);
    std::cout << "[GeoJSON] 加载 " << points.size() << " 个点 from " << geojson_path << "\n";
    return points;
}

// =========================================================================
// 路径评估指标结构体 (用于 FMM / A* / RRT* / FGDA* 四算法对比)
// =========================================================================
struct PathMetrics {
    std::string planner;
    std::string postprocess_mode;
    bool   success;                  // 规划是否成功；失败行的数值指标为 NaN
    int    pair_index;              // 实验编号 (起止点对序号, 0-based)
    int    start_idx;               // 起点在 waypoints 中的索引
    int    goal_idx;                // 终点在 waypoints 中的索引
    double start_x, start_y, start_z;  // 起点坐标
    double goal_x,  goal_y,  goal_z;   // 终点坐标
    size_t points;                  // 路径点数
    double length_m;                // 路径总长度 (m)
    double avg_curvature;           // 平均曲率 (总转角/路径长度, 度/m)
    double curvature_sq_integral;   // 曲率平方积分 (1/m)
    double high_curvature_ratio_pct;// 高曲率段长度占比 (%)
    double postprocess_correction_m;// 后处理累计点位修正量 (m)
    double max_turn_deg;            // 最大转角 (度)
    int    max_consecutive_turns;   // 最大连续急转弯段数 (相邻转角均>30°, 避免单点离群主导)
    double min_esdf_m;              // 最小 ESDF 距离 (m)
    double violation_rate_pct;      // 安全裕度违反率 (%)
    int    penetrations;            // 穿墙事件次数
    double time_ms;                 // 规划耗时 (ms)
    double max_climb_deg;           // 最大爬升角 (度)
    double climb_violation_pct;     // 爬升角违反率 (%)
    double total_climb_m;           // 总爬升高度 (m)
    double total_descent_m;         // 总下降高度 (m)
    double avg_height_m;            // 平均飞行高度 (m)
    double height_var;              // 高度方差
    double mean_headwind_mps;       // 沿路径长度加权平均顶风分量 (m/s, 负=净顺风)
    double mean_wind_speed_mps;     // 沿路径长度加权平均风速 (m/s)
};

// 以固定步长对路径做密集采样 (用于 ESDF 安全性检测)
std::vector<Node> sample_path_dense(const std::vector<Node>& path, double step) {
    std::vector<Node> samples;
    if (path.empty()) return samples;
    samples.push_back(path[0]);
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        const Node& a = path[i];
        const Node& b = path[i + 1];
        double dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
        double seg_len = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (seg_len < 1e-9) continue;
        int n = std::max(1, (int)std::ceil(seg_len / step));
        for (int k = 1; k <= n; ++k) {
            double t = (double)k / n;
            samples.push_back(Node(a.x + t*dx, a.y + t*dy, a.z + t*dz));
        }
    }
    return samples;
}

// 评估路径各项指标
//   dem  : DEM 高程 (风场 AGL 换算用)
//   wind : 风场 (可选, 有效时统计顶风/风速指标; 不影响其余指标)
PathMetrics evaluate_path(const std::vector<Node>& path,
                          SDFImageType::Pointer sdfImage,
                          double safety_margin,
                          double time_ms,
                          const std::string& planner_name_str,
                          int pair_index,
                          int start_idx,
                          int goal_idx,
                          const Node& start_pt,
                          const Node& goal_pt,
                          const DEMData& dem,
                          const WindField* wind = nullptr,
                          double high_curvature_threshold_deg_m = 1.0,
                          double postprocess_correction_m = 0.0,
                          const std::string& postprocess_mode = "raw") {
    PathMetrics m;
    m.planner = planner_name_str;
    m.postprocess_mode = postprocess_mode;
    m.success = path.size() >= 2;
    m.pair_index = pair_index;
    m.start_idx = start_idx;
    m.goal_idx  = goal_idx;
    m.start_x = start_pt.x; m.start_y = start_pt.y; m.start_z = start_pt.z;
    m.goal_x  = goal_pt.x;  m.goal_y  = goal_pt.y;  m.goal_z  = goal_pt.z;
    m.time_ms = time_ms;
    m.points = path.size();
    m.mean_headwind_mps = 0.0;
    m.mean_wind_speed_mps = 0.0;
    m.curvature_sq_integral = 0.0;
    m.high_curvature_ratio_pct = 0.0;
    m.postprocess_correction_m = postprocess_correction_m;
    m.max_consecutive_turns = 0;

    // 规划失败：保留失败状态与耗时，所有路径质量指标写 NaN/-1，不能伪装成零优值。
    if (!m.success) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        m.length_m = nan; m.avg_curvature = nan; m.max_turn_deg = nan;
        m.curvature_sq_integral = nan; m.high_curvature_ratio_pct = nan;
        m.postprocess_correction_m = nan;
        m.max_consecutive_turns = -1;
        m.min_esdf_m = nan; m.violation_rate_pct = nan; m.penetrations = -1;
        m.max_climb_deg = nan; m.climb_violation_pct = nan;
        m.total_climb_m = nan; m.total_descent_m = nan;
        m.avg_height_m = nan; m.height_var = nan;
        m.mean_headwind_mps = nan; m.mean_wind_speed_mps = nan;
        return m;
    }

    // ESDF 查询 lambda
    auto query_esdf = [&](double x, double y, double z) -> double {
        SDFImageType::PointType pt; pt[0]=x; pt[1]=y; pt[2]=z;
        SDFImageType::IndexType idx;
        if (sdfImage->TransformPhysicalPointToIndex(pt, idx)) {
            if (sdfImage->GetBufferedRegion().IsInside(idx))
                return static_cast<double>(sdfImage->GetPixel(idx));
        }
        return -100.0;  // 越界视为障碍物
    };

    // 1. 路径长度 + 爬升指标
    m.length_m = 0;
    m.total_climb_m = 0;
    m.total_descent_m = 0;
    m.max_climb_deg = 0;
    int climb_violation_count = 0;
    int climb_segment_count = 0;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        const Node& a = path[i];
        const Node& b = path[i + 1];
        double dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
        double seg_len = std::sqrt(dx*dx + dy*dy + dz*dz);
        m.length_m += seg_len;
        if (dz > 0) m.total_climb_m += dz;
        else        m.total_descent_m += -dz;
        double horizontal = std::sqrt(dx*dx + dy*dy);
        if (horizontal > 1e-9) {
            double climb_angle = std::atan2(dz, horizontal) * 180.0 / M_PI;
            if (std::fabs(climb_angle) > m.max_climb_deg) m.max_climb_deg = std::fabs(climb_angle);
            if (std::fabs(climb_angle) > 15.0) climb_violation_count++;
            climb_segment_count++;
        }
    }
    m.climb_violation_pct = climb_segment_count > 0 ? 100.0 * climb_violation_count / climb_segment_count : 0;

    // 2. 平滑度: 平均曲率 (总转角 / 路径长度) + 最大转角 + 最大连续急转弯段数
    //    连续急转弯: 相邻中间点转角均 > 30° 的最长连续段 (方案新增指标,
    //    避免单个离群点主导结论)
    double total_curvature = 0;
    double high_curvature_length = 0;
    int consecutive_turns = 0;
    m.max_turn_deg = 0;
    for (size_t i = 1; i + 1 < path.size(); ++i) {
        const Node& a = path[i - 1], b = path[i], c = path[i + 1];
        double v1x = b.x - a.x, v1y = b.y - a.y, v1z = b.z - a.z;
        double v2x = c.x - b.x, v2y = c.y - b.y, v2z = c.z - b.z;
        double v1n = std::sqrt(v1x*v1x + v1y*v1y + v1z*v1z);
        double v2n = std::sqrt(v2x*v2x + v2y*v2y + v2z*v2z);
        if (v1n < 1e-9 || v2n < 1e-9) continue;
        double dot = v1x*v2x + v1y*v2y + v1z*v2z;
        double cos_a = std::clamp(dot / (v1n * v2n), -1.0, 1.0);
        double angle_rad = std::acos(cos_a);
        double angle_deg = angle_rad * 180.0 / M_PI;
        double support_len = 0.5 * (v1n + v2n);
        double curvature_rad_m = angle_rad / support_len;
        double curvature_deg_m = angle_deg / support_len;
        total_curvature += angle_deg;
        m.curvature_sq_integral += curvature_rad_m * curvature_rad_m * support_len;
        if (curvature_deg_m >= high_curvature_threshold_deg_m)
            high_curvature_length += support_len;
        if (angle_deg > m.max_turn_deg) m.max_turn_deg = angle_deg;
        // 连续急转弯段统计 (阈值 30°, 与转角约束上限一致)
        if (angle_deg > 30.0) {
            consecutive_turns++;
            m.max_consecutive_turns = std::max(m.max_consecutive_turns, consecutive_turns);
        } else {
            consecutive_turns = 0;
        }
    }
    m.avg_curvature = (m.length_m > 1e-9) ? (total_curvature / m.length_m) : 0.0;
    m.high_curvature_ratio_pct = (m.length_m > 1e-9)
        ? 100.0 * high_curvature_length / m.length_m : 0.0;

    // 3. 安全性: 密集采样 (1m) 检测 ESDF
    std::vector<Node> samples = sample_path_dense(path, 1.0);
    m.min_esdf_m = 1e30;
    int violation_count = 0;
    m.penetrations = 0;
    bool prev_in_wall = false;
    for (const auto& s : samples) {
        double d = query_esdf(s.x, s.y, s.z);
        if (d < m.min_esdf_m) m.min_esdf_m = d;
        if (d < safety_margin) violation_count++;
        if (d < 0) {
            if (!prev_in_wall) m.penetrations++;  // 新的穿墙事件
            prev_in_wall = true;
        } else {
            prev_in_wall = false;
        }
    }
    m.violation_rate_pct = samples.empty() ? 100 : 100.0 * violation_count / samples.size();
    if (m.min_esdf_m > 1e29) m.min_esdf_m = -1;

    // 4. 高度统计
    double sum_z = 0;
    for (const auto& p : path) sum_z += p.z;
    m.avg_height_m = sum_z / path.size();
    double var_sum = 0;
    for (const auto& p : path) {
        double dz = p.z - m.avg_height_m;
        var_sum += dz * dz;
    }
    m.height_var = path.size() > 1 ? var_sum / path.size() : 0;

    // 5. [新增] 风场指标: 逐段中点查询, 水平长度加权平均
    //    顶风分量 = w · d̂_horiz (负=顺风); 与 A*/RRT* 边代价同口径,
    //    用于横向对比 启用/未启用 风场代价的路径能耗特性.
    if (wind && wind->valid()) {
        double hw_sum = 0.0, ws_sum = 0.0, w_len = 0.0;
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            const Node& a = path[i];
            const Node& b = path[i + 1];
            const double dx = b.x - a.x, dy = b.y - a.y;
            const double dh = std::sqrt(dx * dx + dy * dy);
            if (dh < 1e-9) continue;
            double wu, wv;
            wind->query(0.5 * (a.x + b.x), 0.5 * (a.y + b.y),
                        0.5 * (a.z + b.z), dem, wu, wv);
            hw_sum += ((wu * dx + wv * dy) / dh) * dh;
            ws_sum += std::sqrt(wu * wu + wv * wv) * dh;
            w_len  += dh;
        }
        if (w_len > 1e-9) {
            m.mean_headwind_mps   = hw_sum / w_len;
            m.mean_wind_speed_mps = ws_sum / w_len;
        }
    }

    return m;
}

// RRT* 每次实验多次运行各指标的浮动范围 [min, max] (仅 RRT* 使用)
struct RRTFluctuationRow {
    int         pair_index;
    std::string length_range;       // 路径长度(m)
    std::string curvature_range;    // 平均曲率(度/m)
    std::string turn_range;         // 最大转角(度)
    std::string esdf_range;         // 最小ESDF距离(m)
    std::string time_range;         // 规划耗时(ms)
    std::string height_var_range;   // 高度方差(m²)
};

// 将 RRT* 多次运行浮动范围单独写入 CSV (不改动 path_comparison.csv 列结构)
void write_rrt_fluctuation_csv(const std::vector<RRTFluctuationRow>& rows,
                               const std::string& csv_path) {
    std::ofstream ofs(csv_path);
    if (!ofs) {
        std::cerr << "[CSV] 无法写入: " << csv_path << "\n";
        return;
    }
    ofs << "\xEF\xBB\xBF";
    ofs << "实验编号,路径长度(m)[min，max],平均曲率(度/m)[min，max],最大转角(度)[min，max],"
           "最小ESDF距离(m)[min，max],规划耗时(ms)[min，max],高度方差(m²)[min，max]\n";
    for (const auto& r : rows) {
        ofs << r.pair_index << ","
            << r.length_range << ","
            << r.curvature_range << ","
            << r.turn_range << ","
            << r.esdf_range << ","
            << r.time_range << ","
            << r.height_var_range << "\n";
    }
    ofs.close();
    std::cout << "[CSV] RRT* 浮动范围已导出: " << csv_path << "\n";
}

// 将所有规划器的指标写入 CSV (UTF-8 BOM, 中文表头, 兼容 Excel)
void write_metrics_csv(const std::vector<PathMetrics>& all, const std::string& csv_path) {
    std::ofstream ofs(csv_path);
    if (!ofs) {
        std::cerr << "[CSV] 无法写入: " << csv_path << "\n";
        return;
    }
    // UTF-8 BOM: 让 Excel 正确识别中文编码
    ofs << "\xEF\xBB\xBF";
    ofs << "实验编号,起点序号,终点序号,起点X,起点Y,起点Z,终点X,终点Y,终点Z,规划器,后处理模式,规划状态,路径点数,路径长度(m),"
           "平均曲率(度/m),曲率平方积分(1/m),高曲率段占比(%),后处理修正量(m),最大转角(度),最大连续急转弯段数,"
           "最小ESDF距离(m),安全违反率(%),穿墙次数,规划耗时(ms),最大爬升角(度),"
           "爬升违反率(%),总爬升高度(m),总下降高度(m),平均飞行高度(m),高度方差\n";
    for (const auto& m : all) {
        ofs << m.pair_index << ","
            << m.start_idx << "," << m.goal_idx << ","
            << std::fixed << std::setprecision(3)
            << m.start_x << "," << m.start_y << "," << m.start_z << ","
            << m.goal_x  << "," << m.goal_y  << "," << m.goal_z  << ","
            << m.planner << ","
            << m.postprocess_mode << ","
            << (m.success ? "成功" : "失败") << ","
            << m.points << ","
            << m.length_m << ","
            << m.avg_curvature << ","
            << m.curvature_sq_integral << ","
            << m.high_curvature_ratio_pct << ","
            << m.postprocess_correction_m << ","
            << m.max_turn_deg << ","
            << m.max_consecutive_turns << ","
            << m.min_esdf_m << ","
            << m.violation_rate_pct << ","
            << m.penetrations << ","
            << m.time_ms << ","
            << m.max_climb_deg << ","
            << m.climb_violation_pct << ","
            << m.total_climb_m << ","
            << m.total_descent_m << ","
            << m.avg_height_m << ","
            << m.height_var << "\n";
    }
    ofs.close();
    std::cout << "[CSV] 对比结果已导出: " << csv_path << "\n";
}

// [新增] 将风场指标单独写入 CSV (不改动 path_comparison.csv 的既有列结构,
//        保持下游 Analysis.py 等解析脚本兼容)
void write_wind_metrics_csv(const std::vector<PathMetrics>& all,
                            const std::string& csv_path,
                            bool wind_cost_on) {
    std::ofstream ofs(csv_path);
    if (!ofs) {
        std::cerr << "[CSV] 无法写入: " << csv_path << "\n";
        return;
    }
    ofs << "\xEF\xBB\xBF";
    ofs << "实验编号,规划器,规划状态,平均顶风分量(m/s),平均风速(m/s),风场代价\n";
    for (const auto& m : all) {
        ofs << m.pair_index << "," << m.planner << ","
            << (m.success ? "成功" : "失败") << ","
            << std::fixed << std::setprecision(3)
            << m.mean_headwind_mps << ","
            << m.mean_wind_speed_mps << ","
            << (wind_cost_on ? "on" : "off") << "\n";
    }
    ofs.close();
    std::cout << "[CSV] 风场指标已导出: " << csv_path << "\n";
}

// 控制台打印单组实验规划器对比 (紧凑表格)
void print_pair_summary(int pair_idx, const Node& start, const Node& goal,
                        const std::vector<PathMetrics>& pair_metrics) {
    int si = pair_metrics.empty() ? -1 : pair_metrics.front().start_idx;
    int gi = pair_metrics.empty() ? -1 : pair_metrics.front().goal_idx;
    std::cout << "\n========== 实验 " << pair_idx
              << " [pt" << si << " → pt" << gi << "]"
              << ": (" << start.x << ", " << start.y << ", " << start.z << ")"
              << " → (" << goal.x << ", " << goal.y << ", " << goal.z << ") ==========\n";
    std::cout << std::left;
    std::cout << std::setw(10) << "规划器"
              << std::setw(10) << "点数"
              << std::setw(14) << "长度(m)"
              << std::setw(14) << "min_d(m)"
              << std::setw(10) << "穿墙"
              << std::setw(12) << "耗时(ms)"
              << std::setw(14) << "最大转角"
              << std::setw(14) << "最大爬升" << "\n";
    std::cout << "------------------------------------------------------------------------\n";
    for (const auto& m : pair_metrics) {
        std::cout << std::setw(10) << m.planner
                  << std::setw(10) << (m.success ? "成功" : "失败")
                  << std::setw(10) << m.points
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << m.length_m
                  << std::setw(14) << m.min_esdf_m
                  << std::setw(10) << m.penetrations
                  << std::setw(12) << m.time_ms
                  << std::setw(14) << m.max_turn_deg
                  << std::setw(14) << m.max_climb_deg << "\n";
    }
    std::cout << "========================================================================\n";
}

// 严格解析命令行数值：必须完整消费字符串，拒绝空串、尾随字符、溢出和非有限值。
bool parse_int_strict(const char* text, int& value) {
    if (!text || *text == '\0') return false;
    const char* end = text + std::strlen(text);
    auto result = std::from_chars(text, end, value);
    return result.ec == std::errc() && result.ptr == end;
}

bool parse_double_strict(const char* text, double& value) {
    if (!text || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    value = std::strtod(text, &end);
    return errno != ERANGE && end == text + std::strlen(text) && std::isfinite(value);
}

// =========================================================================
// runPathPlanningComparison
//   加载 ESDF/DEM/航点, 批量运行 FMM/A*/RRT*/FGDA* 四算法对比实验,
//   导出 CSV 指标文件 + 指定起止点对的 shapefile.
//   返回 0 成功, 非 0 失败.
// =========================================================================
int runPathPlanningComparison(int argc, char* argv[]) {
    // ========================= 用户手动配置 =========================
    // 默认依次运行四个 ESDF 规划器，无需手动切换

    // ===== [新增] 风场代价实验配置 (节能航线规划) =====
    // 命令行参数 (均带默认值, 不传参 = 原实验行为):
    //   --wind              启用风场代价 (节能规划), 产物 → data/exp_wind/
    //   --wind-nc <path>    风场 nc 路径 (默认 /workspace/01_uav_path_planning/data/interpolated_wind_20260102_1800_2m.nc)
    //   --lambda-w <val>    顶风惩罚系数 λ_w, 默认 0.05 (5 m/s 顶风 → 代价 +25%)
    //
    // use_wind_cost = false (默认): 完全等同于原实验 (规划器收到 wind=nullptr,
    //                        代价模型逐位一致; 若风场 nc 存在则仅用于
    //                        指标统计, 不影响路径).
    // use_wind_cost = true (--wind): A*/RRT* 边代价 ×(1+λ_w·max(0,顶风分量)),
    //                        FMM 速度场按 |w| 各向同性近似 (Eikonal 限制);
    //                        输出到独立目录 data/exp_wind/, 与原实验产物
    //                        (data/exp_nowind/) 分开管理, 互不覆盖.
    //                        此时风场 nc 必须可加载, 否则直接退出.
    bool        use_wind_cost = false;
    bool        export_fmm_3d = false;
    bool        export_fgda_corridor = false;
    std::string wind_nc_path  = "/workspace/01_uav_path_planning/data/interpolated_wind_20260102_1800_2m.nc";
    double      wind_lambda_w = 0.05;  // 物理上界 ≈ 3/v_g (能量模型线性化),
                                       // 首版取保守值, 后续可升级能量模型
    // [方案B] RRT* 多次运行配置 (回应审稿: 随机采样算法单次结果不可靠,
    //         需多次独立运行取中位数以消除离群值)
    int         rrt_runs  = 10;    // 每组起终点独立运行次数 (按长度中位数取代表)
    int         rrt_seed  = 42;    // 基础随机种子 (第 k 次 = rrt_seed + k)
    int         rrt_iters = 150;   // 采样预算参数 (内部实际 = max(×30, 3000) 次采样)
    FGDAStarConfig fgda_config;
    bool fgda_only = false;
    bool fmm_only = false;
    bool astar_only = false;
    std::string planner_label;
    PostprocessMode postprocess_mode = PostprocessMode::Raw;
    std::string postprocess_mode_name = "raw";
    double high_curvature_threshold = 1.0;
    std::string output_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& option) -> const char* {
            if (i + 1 >= argc || argv[i + 1][0] == '\0' || std::strncmp(argv[i + 1], "--", 2) == 0) {
                std::cerr << "[参数] " << option << " 缺少值\n";
                return nullptr;
            }
            return argv[++i];
        };
        auto parse_int_option = [&](int& target) {
            const char* value = require_value(arg);
            if (!value) return false;
            if (!parse_int_strict(value, target)) {
                std::cerr << "[参数] " << arg << " 的值无效: " << value << "\n";
                return false;
            }
            return true;
        };
        auto parse_double_option = [&](double& target) {
            const char* value = require_value(arg);
            if (!value) return false;
            if (!parse_double_strict(value, target)) {
                std::cerr << "[参数] " << arg << " 的值无效: " << value << "\n";
                return false;
            }
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            std::cout << "用法: uav_planner [--fgda-only|--fmm-only|--astar-only] "
                      << "[--postprocess-mode raw|smoothed]\n"
                      << "  --fgda-position-state          使用位置状态的 FMM 启发 A*\n"
                      << "  --planner-label <label>        单算法实验 CSV 标签\n"
                      << "  --rrt-runs <n> --rrt-seed <n> --rrt-iters <n>\n"
                      << "  --export-fmm-3d              导出完整 FMM 三维到达时间场\n"
                      << "  --export-fgda-corridor       导出 FDGA* 引导路径、走廊和到达代价场\n"
                      << "  --output-dir <path>\n";
            return 0;
        }
        else if (arg == "--wind") use_wind_cost = true;
        else if (arg == "--wind-nc") {
            const char* value = require_value(arg); if (!value) return -1; wind_nc_path = value;
        }
        else if (arg == "--lambda-w") { if (!parse_double_option(wind_lambda_w)) return -1; }
        else if (arg == "--rrt-runs") { if (!parse_int_option(rrt_runs)) return -1; }
        else if (arg == "--rrt-seed") { if (!parse_int_option(rrt_seed)) return -1; }
        else if (arg == "--rrt-iters") { if (!parse_int_option(rrt_iters)) return -1; }
        else if (arg == "--fgda-curvature-weight") { if (!parse_double_option(fgda_config.curvature_weight)) return -1; }
        else if (arg == "--fgda-traversal-weight") { if (!parse_double_option(fgda_config.traversal_weight)) return -1; }
        else if (arg == "--fgda-clearance-weight") { if (!parse_double_option(fgda_config.clearance_weight)) return -1; }
        else if (arg == "--fgda-heuristic-weight") { if (!parse_double_option(fgda_config.heuristic_weight)) return -1; }
        else if (arg == "--fgda-max-turn-angle") { if (!parse_double_option(fgda_config.max_turn_angle_deg)) return -1; }
        else if (arg == "--fgda-corridor-radius") { if (!parse_int_option(fgda_config.corridor_radius)) return -1; }
        else if (arg == "--fgda-position-state") fgda_config.position_state = true;
        else if (arg == "--planner-label") {
            const char* value = require_value(arg); if (!value) return -1; planner_label = value;
        }
        else if (arg == "--postprocess-mode") {
            const char* value = require_value(arg); if (!value) return -1;
            if (std::strcmp(value, "raw") == 0) { postprocess_mode = PostprocessMode::Raw; postprocess_mode_name = "raw"; }
            else if (std::strcmp(value, "smoothed") == 0) { postprocess_mode = PostprocessMode::Smoothed; postprocess_mode_name = "smoothed"; }
            else { std::cerr << "[参数] --postprocess-mode 必须为 raw 或 smoothed\n"; return -1; }
        }
        else if (arg == "--fgda-only") fgda_only = true;
        else if (arg == "--fmm-only") fmm_only = true;
        else if (arg == "--astar-only") astar_only = true;
        else if (arg == "--export-fmm-3d") export_fmm_3d = true;
        else if (arg == "--export-fgda-corridor") export_fgda_corridor = true;
        else if (arg == "--high-curvature-threshold") {
            if (!parse_double_option(high_curvature_threshold)) return -1;
        }
        else if (arg == "--output-dir") {
            const char* value = require_value(arg); if (!value) return -1; output_dir = value;
        }
        else {
            std::cerr << "[参数] 未知参数: " << arg << "\n"
                      << "用法: uav_planner [--wind] [--wind-nc <path>] [--lambda-w <val>]\n"
                      << "                [--rrt-runs <n>] [--rrt-seed <n>] [--rrt-iters <n>]\n"
                      << "                [--fgda-only|--fmm-only|--astar-only] [--postprocess-mode raw|smoothed]\n"
                      << "                [--fgda-curvature-weight <val>] [--fgda-traversal-weight <val>]\n"
                      << "                [--fgda-clearance-weight <val>] [--fgda-heuristic-weight <val>]\n"
                      << "                [--fgda-max-turn-angle <deg>] [--fgda-corridor-radius <cells>]\n"
                      << "                [--fgda-position-state] [--planner-label <label>]\n"
                      << "                [--high-curvature-threshold <deg/m>] [--export-fmm-3d]\n"
                      << "                [--export-fgda-corridor] [--output-dir <path>]\n";
            return -1;
        }
    }
    if (rrt_runs < 1 || rrt_iters < 1 || wind_lambda_w < 0.0 ||
        fgda_config.curvature_weight < 0.0 || fgda_config.traversal_weight < 0.0 ||
        fgda_config.clearance_weight < 0.0 || fgda_config.heuristic_weight < 0.0 ||
        fgda_config.corridor_radius < 0 || high_curvature_threshold < 0.0 ||
        (fgda_config.max_turn_angle_deg != -1.0 &&
         (fgda_config.max_turn_angle_deg <= 0.0 || fgda_config.max_turn_angle_deg > 180.0)) ||
        static_cast<int>(fgda_only) + static_cast<int>(fmm_only) + static_cast<int>(astar_only) > 1 ||
        (export_fmm_3d && export_fgda_corridor) ||
        (!planner_label.empty() && !(fgda_only || fmm_only || astar_only))) {
        std::cerr << "[参数] 运行次数须为正数，FGDA* 权重、走廊半径及高曲率阈值不得为负数；"
                  << "最大转角须在 (0,180] 或为 -1，且 only 选项不可组合。\n";
        return -1;
    }
    // ================================================================

    std::vector<PlannerType> planners_to_run;
    if (export_fmm_3d) planners_to_run = {FMM_PLANNER};
    else if (export_fgda_corridor) planners_to_run = {FGDASTAR_PLANNER};
    else if (fgda_only) planners_to_run = {FGDASTAR_PLANNER};
    else if (fmm_only) planners_to_run = {FMM_PLANNER};
    else if (astar_only) planners_to_run = {ASTAR_PLANNER};
    else planners_to_run = {FGDASTAR_PLANNER, FMM_PLANNER, ASTAR_PLANNER, RRT_STAR_PLANNER};

        auto planner_name = [&](PlannerType p) -> std::string {
            if (!planner_label.empty()) return planner_label;
            switch (p) {
                case FGDASTAR_PLANNER:return "FGDA*";
                case FMM_PLANNER:     return "FMM";
                case ASTAR_PLANNER:   return "A*";
                case RRT_STAR_PLANNER:return "RRT*";
                default:              return "UNKNOWN";
            }
        };
        std::cout << "碰撞检测方式: ESDF" << std::endl;
        std::cout << "本次将依次运行以下规划器: ";
        for (auto p : planners_to_run) std::cout << planner_name(p) << " ";
        std::cout << std::endl;

        const float safety_margin = 5.0f;
        // 统一规划超时阈值: 1 分钟 (单次规划调用口径; RRT* 每次 run 独立计时)
        constexpr double PLANNER_TIMEOUT_SECONDS = 60.0;

        // 唯一需要保存 shapefile 的起止点对 (其余实验只收集指标, 不保存 shp)
        const Node shp_save_start(51913.631, 27162.232, 15.0614);
        const Node shp_save_goal (53056.661, 28918.611, 15.087);

        // ---------- 1. 加载所需数据 (ESDF, DEM, 航点 GeoJSON) ----------
        SDFImageType::Pointer sdfImage = nullptr;
        DEMData dem;

        // 1.1 加载 ESDF
        const std::string sdf_cache_file = "/workspace/01_uav_path_planning/data/SF_Downtown_sdf.mhd";
        if (file_exists(sdf_cache_file)) {
            std::cout << "[ESDF] 检测到缓存文件，加载中...\n";
            using ReaderType = itk::ImageFileReader<SDFImageType>;
            auto reader = ReaderType::New();
            reader->SetFileName(sdf_cache_file);
            try {
                reader->Update();
                sdfImage = reader->GetOutput();
                std::cout << "[ESDF] 加载成功！\n";
            } catch (const itk::ExceptionObject& e) {
                std::cerr << "[ESDF] 加载失败: " << e.GetDescription() << std::endl;
                sdfImage = nullptr;
            }
        }
        if (!sdfImage) {
            std::cerr << "[错误] 无有效 SDF 缓存，安全距离判定将无法进行！\n";
            return -1;
        }

        // 1.2 加载 DEM
        const std::string dem_file = "/workspace/01_uav_path_planning/data/SF_Downtown.tif";
        if (!loadDEMDataFromFile(dem_file, dem)) {
            std::cerr << "[DEM] 高程图加载失败！\n";
            return -1;
        }
        std::cout << "[DEM] 加载成功，尺寸 " << dem.width << "x" << dem.height << "\n";

        // 1.2.5 [新增] 加载风场
        //   use_wind_cost=true : 必须成功 (节能规划依赖), 失败即退出;
        //   use_wind_cost=false: 文件存在则加载 (仅指标统计, 路径不受影响),
        //                        不存在则纯原模式.
        WindField wind;
        bool wind_loaded = false;
        if (use_wind_cost || file_exists(wind_nc_path)) {
            wind_loaded = wind.load(wind_nc_path, wind_lambda_w);
            if (use_wind_cost && !wind_loaded) {
                std::cerr << "[错误] use_wind_cost=true 但风场加载失败: "
                          << wind_nc_path << "\n";
                return -1;
            }
        }
        const WindField* wind_for_cost = (use_wind_cost && wind_loaded) ? &wind : nullptr;
        std::cout << "[Wind] 风场代价: " << (wind_for_cost ? "启用 (节能规划)"
                                                            : "关闭")
                  << (wind_loaded ? " | 风场指标统计: 启用" : "")
                  << "\n";
        const std::string out_dir = output_dir.empty()
            ? "/workspace/01_uav_path_planning/data/exp_" + postprocess_mode_name + (use_wind_cost ? "_wind" : "")
            : output_dir;
        std::error_code out_ec;
        std::filesystem::create_directories(out_dir, out_ec);
        if (out_ec) {
            std::cerr << "[Out] 无法创建输出目录 " << out_dir << ": " << out_ec.message() << "\n";
            return -1;
        }
        std::cout << "[Out] 实验产物目录: " << out_dir << "\n";

        // 1.3 加载航点 GeoJSON (3D 点, 两两全排列有向对: pt_i → pt_j, i≠j)
        const std::string geojson_path = "/workspace/01_uav_path_planning/data/points_3d.geojson";
        std::vector<Node> waypoints = load_points_from_geojson(geojson_path);
        if (waypoints.size() < 2) {
            std::cerr << "[错误] 航点不足 2 个, 无法组成起止点对 (现有 "
                      << waypoints.size() << ")\n";
            return -1;
        }
        const int n_pts = (int)waypoints.size();
        const int num_pairs = n_pts * (n_pts - 1);   // 全排列: A(n,2) = n*(n-1), 含起止点互换
        constexpr int EXPECTED_PAIRS = 42;
        if (num_pairs != EXPECTED_PAIRS) {
            std::cerr << "[错误] 实验配置校验失败: 期望 42 组有向起止点对（7 个航点），实际 "
                      << num_pairs << " 组（" << n_pts << " 个航点）\n";
            delete[] dem.data;
            dem.data = nullptr;
            return -1;
        }
        std::cout << "[Batch] 42 组实验配置校验通过。共 " << n_pts << " 个航点, "
                  << num_pairs << " 组有向起止点对 (含互换), 每组跑 "
                  << planners_to_run.size() << " 个规划器\n";

        // ---------- 2. 批量实验主循环 (全排列有向对) ----------
        std::vector<PathMetrics> all_metrics;
        std::vector<RRTFluctuationRow> rrt_fluctuation;  // RRT* 每组多次运行的指标浮动范围
        auto match_node = [](const Node& a, const Node& b, double tol = 1.0) {
            return std::abs(a.x - b.x) < tol && std::abs(a.y - b.y) < tol && std::abs(a.z - b.z) < tol;
        };

        int pair_idx = 0;
        for (int i = 0; i < n_pts; ++i) {
            for (int j = 0; j < n_pts; ++j) {
                if (i == j) continue;   // 起止不能相同

                const Node& start_pt = waypoints[i];
                const Node& goal_pt  = waypoints[j];
                bool is_shp_pair = match_node(start_pt, shp_save_start) && match_node(goal_pt, shp_save_goal);
                if ((export_fmm_3d || export_fgda_corridor) && !is_shp_pair) continue;
#if ONLY_SHP_PAIR
                if (!is_shp_pair) continue;   // 仅运行 shp_save 起终点组, 跳过其余实验
#endif

                std::cout << "\n############## 实验 " << pair_idx << "/" << (num_pairs - 1)
                          << " [pt" << i << " → pt" << j << "] ##############\n";
                std::cout << "起点: (" << start_pt.x << ", " << start_pt.y << ", " << start_pt.z << ")\n";
                std::cout << "终点: (" << goal_pt.x  << ", " << goal_pt.y  << ", " << goal_pt.z  << ")\n";
                if (is_shp_pair) std::cout << "[ShpSave] 此组将保存 shapefile\n";

                std::vector<PathMetrics> pair_metrics;

                for (PlannerType pt : planners_to_run) {
                    std::cout << "\n---------- [" << planner_name(pt) << "] 开始规划 ----------\n";
                    std::cout.flush();

                    // 计时区静默 std::cout: 规划器内部有大量诊断打印 (FMM 进度/RRT* iter 等),
                    // 终端 I/O 刷新会污染耗时. 置 badbit 让所有 << 操作直接跳过格式化返回,
                    // 连格式化开销也省去, 保证耗时反映算法本身.
                    std::cout.setstate(std::ios::badbit);

                    std::vector<Node> final_path;
                    double postprocess_correction_m = 0.0;
                    std::string failure_reason;   // 规划器失败时回填可定位原因
                    // 搜索过程数据 (仅 is_shp_pair 时导出, 用于可视化)
                    FMMSearchData fmm_search;
                    FGDAStarSearchData fgda_search;
                    AStarSearchData astar_search;
                    RRTSearchData rrt_search;

                    // 单调时钟避免系统时间校准影响规划耗时。
                    using Clock = std::chrono::steady_clock;
                    double duration_ms = 0.0;

                    switch (pt) {
                        case FGDASTAR_PLANNER: {
                            auto t0 = Clock::now();
                            fgda_config.timeout_seconds = export_fgda_corridor
                                ? std::numeric_limits<double>::infinity()
                                : PLANNER_TIMEOUT_SECONDS;
                            final_path = plan_fgdastar_trajectory(
                                start_pt, goal_pt, sdfImage, dem, safety_margin, fgda_config,
                                wind_for_cost, is_shp_pair ? &fgda_search : nullptr,
                                &failure_reason);
                            auto t1 = Clock::now();
                            duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                            break;
                        }
                        case FMM_PLANNER: {
                            auto t0 = Clock::now();
                            final_path = plan_fmm_trajectory(start_pt, goal_pt, sdfImage, dem,
                                                             150, 25, safety_margin, 0.05,
                                                             is_shp_pair ? &fmm_search : nullptr,
                                                             wind_for_cost,
                                                             export_fmm_3d
                                                                 ? std::numeric_limits<double>::infinity()
                                                                 : PLANNER_TIMEOUT_SECONDS,
                                                             export_fmm_3d);
                            auto t1 = Clock::now();
                            duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                            break;
                        }
                        case ASTAR_PLANNER: {
                            auto t0 = Clock::now();
                            final_path = plan_astar_trajectory(start_pt, goal_pt, sdfImage, dem,
                                                               150, 25, safety_margin, 0.05,
                                                               is_shp_pair ? &astar_search : nullptr,
                                                               wind_for_cost,
                                                               PLANNER_TIMEOUT_SECONDS);
                            auto t1 = Clock::now();
                            duration_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                            break;
                        }
                        case RRT_STAR_PLANNER: {
                            // [方案B] 多次独立运行 (不同 seed), 按 3D 路径长度取中位数代表,
                            //         消除随机采样的单次波动与离群值.
                            auto path_len = [](const std::vector<Node>& p) {
                                double L = 0.0;
                                for (size_t i = 1; i < p.size(); ++i)
                                    L += get_distance(p[i - 1], p[i]);
                                return L;
                            };
                            std::vector<std::vector<Node>> run_paths;
                            std::vector<RRTSearchData> run_trees;   // 仅 is_shp_pair 时用
                            std::vector<double> run_ms;
                            std::vector<double> run_corrections;
                            std::vector<PathMetrics> run_metrics;   // 每次有效运行的完整指标 (浮动范围统计)
                            run_paths.reserve(rrt_runs); run_ms.reserve(rrt_runs);
                            run_corrections.reserve(rrt_runs); run_metrics.reserve(rrt_runs);
                            if (is_shp_pair) run_trees.reserve(rrt_runs);
                            double total_ms = 0.0;
                            for (int k = 0; k < rrt_runs; ++k) {
                                RRTSearchData tree_k;
                                auto t0 = Clock::now();
                                auto p = plan_rrt_star_trajectory(
                                    start_pt, goal_pt, sdfImage, dem,
                                    rrt_iters, 25, safety_margin, 0.05,
                                    is_shp_pair ? &tree_k : nullptr,
                                    wind_for_cost, (unsigned int)(rrt_seed + k),
                                    PLANNER_TIMEOUT_SECONDS);
                                auto t1 = Clock::now();
                                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                                total_ms += ms;
                                if (!p.empty()) {
                                    double run_correction = 0.0;
                                    p = postprocess_node_path(
                                        p, sdfImage, dem, safety_margin,
                                        postprocess_mode, &run_correction);
                                    run_paths.push_back(std::move(p));
                                    run_ms.push_back(ms);
                                    run_corrections.push_back(run_correction);
                                    if (is_shp_pair) run_trees.push_back(std::move(tree_k));
                                    // 中位数选择与浮动范围均使用当前统一后处理模式下的路径。
                                    run_metrics.push_back(evaluate_path(
                                        run_paths.back(), sdfImage, safety_margin, ms,
                                        "RRT*", pair_idx, i, j, start_pt, goal_pt, dem,
                                        wind_loaded ? &wind : nullptr,
                                        high_curvature_threshold, run_correction,
                                        postprocess_mode_name));
                                }
                            }
                            if (!run_paths.empty()) {
                                std::vector<size_t> order(run_paths.size());
                                std::iota(order.begin(), order.end(), 0);
                                std::sort(order.begin(), order.end(),
                                          [&](size_t a, size_t b) {
                                              return path_len(run_paths[a]) < path_len(run_paths[b]);
                                          });
                                size_t mid = order[order.size() / 2];
                                final_path = run_paths[mid];
                                postprocess_correction_m = run_corrections[mid];
                                duration_ms = run_ms[mid];   // 代表路径单次耗时 (公平口径)
                                if (is_shp_pair && mid < run_trees.size()) rrt_search = run_trees[mid];
                                // [方案B] 统计本次多次运行各指标的浮动范围 [min, max]
                                auto fmt_range = [](double mn, double mx) {
                                    std::ostringstream oss;
                                    oss << std::fixed << std::setprecision(3)
                                        << "[" << mn << "，" << mx << "]";
                                    return oss.str();
                                };
                                auto range_of = [&](double PathMetrics::*f) {
                                    double mn = 1e30, mx = -1e30;
                                    for (const auto& rm : run_metrics) {
                                        double v = rm.*f;
                                        if (v < mn) mn = v;
                                        if (v > mx) mx = v;
                                    }
                                    return fmt_range(mn, mx);
                                };
                                RRTFluctuationRow row;
                                row.pair_index       = pair_idx;
                                row.length_range     = range_of(&PathMetrics::length_m);
                                row.curvature_range  = range_of(&PathMetrics::avg_curvature);
                                row.turn_range       = range_of(&PathMetrics::max_turn_deg);
                                row.esdf_range       = range_of(&PathMetrics::min_esdf_m);
                                row.time_range       = range_of(&PathMetrics::time_ms);
                                row.height_var_range = range_of(&PathMetrics::height_var);
                                rrt_fluctuation.push_back(std::move(row));
                            } else {
                                // 全部轮次失败/超时: 耗时记录 N 次运行总耗时, 反映真实计算成本
                                duration_ms = total_ms;
                            }
                            // 采样预算说明: 内部实际 = max(rrt_iters*30, 3000)
                            std::cout << "[RRT*] 多run=" << rrt_runs << "/有效=" << run_paths.size()
                                      << ", 采样预算=" << std::max(rrt_iters * 30, 3000)
                                      << ", 代表(长度中位数)耗时=" << duration_ms
                                      << " ms, N次总耗时=" << total_ms << " ms\n";
                            break;
                        }
                        default:
                            std::cerr << "未知规划器类型！\n";
                            return -1;
                    }

                    if (pt != RRT_STAR_PLANNER && !export_fmm_3d && !export_fgda_corridor) {
                        final_path = postprocess_node_path(
                            final_path, sdfImage, dem, safety_margin,
                            postprocess_mode, &postprocess_correction_m);
                    }
                    std::cout.clear();
                    if (export_fmm_3d) {
                        if (fmm_search.T.empty()) {
                            std::cerr << "[FMM 3D] 到达时间场为空，无法导出。\n";
                            delete[] dem.data;
                            dem.data = nullptr;
                            return -1;
                        }

                        using ArrivalImageType = itk::Image<float, 3>;
                        ArrivalImageType::SizeType volume_size;
                        volume_size[0] = static_cast<ArrivalImageType::SizeType::SizeValueType>(fmm_search.nx);
                        volume_size[1] = static_cast<ArrivalImageType::SizeType::SizeValueType>(fmm_search.ny);
                        volume_size[2] = static_cast<ArrivalImageType::SizeType::SizeValueType>(fmm_search.nz);

                        ArrivalImageType::IndexType volume_start;
                        volume_start.Fill(0);
                        ArrivalImageType::RegionType volume_region;
                        volume_region.SetIndex(volume_start);
                        volume_region.SetSize(volume_size);

                        auto arrival_image = ArrivalImageType::New();
                        arrival_image->SetRegions(volume_region);
                        ArrivalImageType::SpacingType volume_spacing;
                        volume_spacing.Fill(fmm_search.ds);
                        arrival_image->SetSpacing(volume_spacing);
                        ArrivalImageType::PointType volume_origin;
                        volume_origin[0] = fmm_search.bx_min;
                        volume_origin[1] = fmm_search.by_min;
                        volume_origin[2] = fmm_search.bz_min;
                        arrival_image->SetOrigin(volume_origin);
                        arrival_image->Allocate();
                        std::copy(fmm_search.T.begin(), fmm_search.T.end(),
                                  arrival_image->GetBufferPointer());

                        const std::string volume_path =
                            out_dir + "/search_FMM_Tfield_3D.mhd";
                        auto arrival_writer = itk::ImageFileWriter<ArrivalImageType>::New();
                        arrival_writer->SetFileName(volume_path);
                        arrival_writer->SetInput(arrival_image);
                        arrival_writer->SetUseCompression(false);
                        try {
                            arrival_writer->Update();
                        } catch (const itk::ExceptionObject& error) {
                            std::cerr << "[FMM 3D] 保存失败: " << error << "\n";
                            delete[] dem.data;
                            dem.data = nullptr;
                            return -1;
                        }

                        std::cout << "[FMM 3D] 已保存完整三维到达时间场: "
                                  << volume_path << " ("
                                  << fmm_search.nx << "x"
                                  << fmm_search.ny << "x"
                                  << fmm_search.nz << ")\n";
                        delete[] dem.data;
                        dem.data = nullptr;
                        return 0;
                    }

                    if (export_fgda_corridor) {
                        if (final_path.empty() || fgda_search.fmm.T.empty() ||
                            fgda_search.corridor.empty() || fgda_search.guidance_path.empty()) {
                            std::cerr << "[FDGA*走廊] 规划或走廊构建失败，无法导出。原因: "
                                      << failure_reason << "\n";
                            delete[] dem.data;
                            dem.data = nullptr;
                            return -1;
                        }

                        const auto& field = fgda_search.fmm;
                        auto write_mhd = [&](const std::string& stem,
                                             const std::string& element_type,
                                             const char* buffer,
                                             size_t byte_count) {
                            const std::string raw_path = out_dir + "/" + stem + ".raw";
                            std::ofstream raw(raw_path, std::ios::binary);
                            raw.write(buffer, static_cast<std::streamsize>(byte_count));
                            if (!raw) return false;
                            raw.close();

                            std::ofstream mhd(out_dir + "/" + stem + ".mhd");
                            mhd << std::setprecision(17)
                                << "ObjectType = Image\n"
                                << "NDims = 3\n"
                                << "BinaryData = True\n"
                                << "BinaryDataByteOrderMSB = False\n"
                                << "CompressedData = False\n"
                                << "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
                                << "Offset = " << field.bx_min << " " << field.by_min
                                << " " << field.bz_min << "\n"
                                << "CenterOfRotation = 0 0 0\n"
                                << "AnatomicalOrientation = RAI\n"
                                << "ElementSpacing = " << field.ds << " " << field.ds
                                << " " << field.ds << "\n"
                                << "DimSize = " << field.nx << " " << field.ny
                                << " " << field.nz << "\n"
                                << "ElementType = " << element_type << "\n"
                                << "ElementDataFile = " << stem << ".raw\n";
                            return static_cast<bool>(mhd);
                        };

                        const bool field_ok = write_mhd(
                            "search_FGDASTAR_arrival_cost_3D", "MET_FLOAT",
                            reinterpret_cast<const char*>(field.T.data()),
                            field.T.size() * sizeof(float));
                        const bool corridor_ok = write_mhd(
                            "search_FGDASTAR_corridor", "MET_UCHAR",
                            reinterpret_cast<const char*>(fgda_search.corridor.data()),
                            fgda_search.corridor.size() * sizeof(unsigned char));

                        std::ofstream guidance(
                            out_dir + "/search_FGDASTAR_guidance_path.csv");
                        guidance << "step,ix,iy,iz,x,y,z,T\n" << std::setprecision(9);
                        for (size_t k = 0; k < fgda_search.guidance_path.size(); ++k) {
                            const Node& p = fgda_search.guidance_path[k];
                            const int ix = static_cast<int>(std::round(
                                (p.x - field.bx_min) / field.ds));
                            const int iy = static_cast<int>(std::round(
                                (p.y - field.by_min) / field.ds));
                            const int iz = static_cast<int>(std::round(
                                (p.z - field.bz_min) / field.ds));
                            guidance << k << "," << ix << "," << iy << "," << iz
                                     << "," << p.x << "," << p.y << "," << p.z
                                     << "," << fgda_search.guidance_arrival[k] << "\n";
                        }

                        std::ofstream path_csv(
                            out_dir + "/search_path_FGDASTAR_raw.csv");
                        path_csv << "x,y,z\n" << std::setprecision(9);
                        for (const Node& p : final_path)
                            path_csv << p.x << "," << p.y << "," << p.z << "\n";

                        if (!field_ok || !corridor_ok || !guidance || !path_csv) {
                            std::cerr << "[FDGA*走廊] 文件写入失败。\n";
                            delete[] dem.data;
                            dem.data = nullptr;
                            return -1;
                        }
                        std::cout << "[FDGA*走廊] 已导出引导路径、走廊掩膜、"
                                  << "到达代价场和原始规划路径: " << out_dir << "\n";
                        delete[] dem.data;
                        dem.data = nullptr;
                        return 0;
                    }

                    if (final_path.empty()) {
                        if (failure_reason.empty()) failure_reason = "规划器返回空路径（未提供内部诊断）";
                        std::cerr << "[规划失败] 实验=" << pair_idx
                                  << ", 航点=pt" << i << "→pt" << j
                                  << ", 规划器=" << planner_name(pt)
                                  << ", 原因=" << failure_reason << "\n";
                        if (pt == FGDASTAR_PLANNER) {
                            std::cerr << "[FGDA*配置] 走廊半径=" << fgda_config.corridor_radius
                                      << "格, 最大转角=" << fgda_config.max_turn_angle_deg
                                      << "°, 曲率权重=" << fgda_config.curvature_weight
                                      << ", 位置状态=" << (fgda_config.position_state ? "是" : "否")
                                      << "\n";
                        }
                    }
                    std::cout << "[" << planner_name(pt) << "] 耗时 " << duration_ms << " ms\n";

                    // 评估指标 (风场可用时附带统计顶风/风速, 不影响路径)
                    PathMetrics pm = evaluate_path(final_path, sdfImage, safety_margin,
                                                    duration_ms, planner_name(pt),
                                                    pair_idx, i, j, start_pt, goal_pt,
                                                    dem, wind_loaded ? &wind : nullptr,
                                                    high_curvature_threshold,
                                                    postprocess_correction_m,
                                                    postprocess_mode_name);
                    all_metrics.push_back(pm);
                    pair_metrics.push_back(pm);

                    std::cout << "[" << planner_name(pt) << "] 指标: "
                              << "点数=" << pm.points
                              << ", 长度=" << pm.length_m << " m"
                              << ", min_d=" << pm.min_esdf_m << " m"
                              << ", 穿墙=" << pm.penetrations
                              << ", 曲率平方积分=" << pm.curvature_sq_integral << " 1/m"
                              << ", 高曲率段占比=" << pm.high_curvature_ratio_pct << "%"
                              << ", 后处理修正量=" << pm.postprocess_correction_m << " m"
                              << ", 最大转角=" << pm.max_turn_deg << "°"
                              << ", 最大连续急转弯段数=" << pm.max_consecutive_turns
                              << ", 最大爬升=" << pm.max_climb_deg << "°"
                              << "\n";
                    if (wind_loaded) {
                        std::cout << "[" << planner_name(pt) << "] 风场: "
                                  << "平均顶风=" << pm.mean_headwind_mps << " m/s"
                                  << " (负=净顺风), 平均风速="
                                  << pm.mean_wind_speed_mps << " m/s"
                                  << (wind_for_cost ? " [代价已启用]" : "")
                                  << "\n";
                    }

                    // 仅对指定起止点对保存 shapefile + 搜索过程数据
                    if (is_shp_pair && !final_path.empty()) {
                        std::string planner_str;
                        switch (pt) {
                            case FGDASTAR_PLANNER: planner_str = "FGDASTAR"; break;
                            case FMM_PLANNER:      planner_str = "FMM"; break;
                            case ASTAR_PLANNER:    planner_str = "ASTAR"; break;
                            case RRT_STAR_PLANNER: planner_str = "RRTSTAR"; break;
                            default:               planner_str = "UNKNOWN"; break;
                        }
                        std::string shp_path = out_dir + "/path_ESDF_" + planner_str + ".shp";
                        save_path_to_shapefile(final_path, shp_path);
                        std::cout << "[ShpSave] 已保存: " << shp_path << "\n";

                        // ===== 保存搜索过程数据 (用于可视化, 同目录管理) =====
                        const std::string search_dir = out_dir + "/";

                        // 最终路径 CSV (x,y,z)
                        {
                            std::ofstream ofs(search_dir + "search_path_" + planner_str + ".csv");
                            ofs << "x,y,z\n" << std::setprecision(6);
                            for (const auto& p : final_path)
                                ofs << p.x << "," << p.y << "," << p.z << "\n";
                            std::cout << "[ShpSave] 已保存路径: search_path_" << planner_str << ".csv\n";
                        }

                        // FMM: 保存二维切片供搜索过程绘图使用
                        if (pt == FMM_PLANNER && !fmm_search.T.empty()) {
                            const int z_target = std::max(0, std::min(fmm_search.nz - 1,
                                static_cast<int>(std::round(
                                    (start_pt.z - fmm_search.bz_min) / fmm_search.ds))));
                            std::ofstream ofs(search_dir + "search_FMM_Tfield.csv");
                            ofs << "x,y,T\n" << std::setprecision(6);
                            for (int y = 0; y < fmm_search.ny; ++y) {
                                for (int x = 0; x < fmm_search.nx; ++x) {
                                    const float tval = fmm_search.T[
                                        (static_cast<size_t>(z_target) * fmm_search.ny + y)
                                        * fmm_search.nx + x];
                                    const double px = fmm_search.bx_min + x * fmm_search.ds;
                                    const double py = fmm_search.by_min + y * fmm_search.ds;
                                    if (std::isinf(tval)) ofs << px << "," << py << ",inf\n";
                                    else ofs << px << "," << py << "," << tval << "\n";
                                }
                            }
                            std::cout << "[ShpSave] 已保存 FMM T 场切片 (Z层=" << z_target
                                      << "): search_FMM_Tfield.csv\n";
                        }

                        // FGDA*: 方向状态 A* 搜索节点 CSV (x,y,z,parent_idx)
                        if (pt == FGDASTAR_PLANNER && !fgda_search.node_x.empty()) {
                            std::ofstream ofs(search_dir + "search_FGDASTAR_nodes.csv");
                            ofs << "x,y,z,parent_idx\n" << std::setprecision(6);
                            for (size_t i = 0; i < fgda_search.node_x.size(); ++i) {
                                ofs << fgda_search.node_x[i] << ","
                                    << fgda_search.node_y[i] << ","
                                    << fgda_search.node_z[i] << ","
                                    << fgda_search.parent_idx[i] << "\n";
                            }
                            std::cout << "[ShpSave] 已保存 FGDA* 搜索节点 ("
                                      << fgda_search.node_x.size()
                                      << " 个): search_FGDASTAR_nodes.csv\n";
                        }

                        // A*: 搜索节点 CSV (x,y,z,parent_idx)
                        if (pt == ASTAR_PLANNER && !astar_search.node_x.empty()) {
                            std::ofstream ofs(search_dir + "search_ASTAR_nodes.csv");
                            ofs << "x,y,z,parent_idx\n" << std::setprecision(6);
                            for (size_t i = 0; i < astar_search.node_x.size(); ++i) {
                                ofs << astar_search.node_x[i] << ","
                                    << astar_search.node_y[i] << ","
                                    << astar_search.node_z[i] << ","
                                    << astar_search.parent_idx[i] << "\n";
                            }
                            std::cout << "[ShpSave] 已保存 A* 搜索节点 ("
                                      << astar_search.node_x.size() << " 个): search_ASTAR_nodes.csv\n";
                        }

                        // RRT*: 搜索树 CSV (x,y,z,parent_idx,cost)
                        if (pt == RRT_STAR_PLANNER && !rrt_search.node_x.empty()) {
                            std::ofstream ofs(search_dir + "search_RRTSTAR_tree.csv");
                            ofs << "x,y,z,parent_idx,cost\n" << std::setprecision(6);
                            for (size_t i = 0; i < rrt_search.node_x.size(); ++i) {
                                ofs << rrt_search.node_x[i] << ","
                                    << rrt_search.node_y[i] << ","
                                    << rrt_search.node_z[i] << ","
                                    << rrt_search.parent_idx[i] << ","
                                    << rrt_search.cost[i] << "\n";
                            }
                            std::cout << "[ShpSave] 已保存 RRT* 搜索树 ("
                                      << rrt_search.node_x.size() << " 个节点): search_RRTSTAR_tree.csv\n";
                        }
                    }
                }

                // 本组实验规划器对比汇总
                print_pair_summary(pair_idx, start_pt, goal_pt, pair_metrics);
                ++pair_idx;
            }
        }

        // ---------- 3. 全量 CSV 导出 ----------
        if (pair_idx != EXPECTED_PAIRS) {
            std::cerr << "[错误] 实验执行数量校验失败: 期望 42 组，实际 " << pair_idx << " 组\n";
            delete[] dem.data;
            dem.data = nullptr;
            return -1;
        }
        const size_t expected_rows = static_cast<size_t>(EXPECTED_PAIRS) * planners_to_run.size();
        if (all_metrics.size() != expected_rows) {
            std::cerr << "[错误] CSV 记录数量校验失败: 期望 " << expected_rows
                      << " 条，实际 " << all_metrics.size() << " 条\n";
            delete[] dem.data;
            dem.data = nullptr;
            return -1;
        }
        std::cout << "[Batch] 42 组执行及 CSV 记录数量校验通过\n";

        //   两类实验分别写入 exp_nowind/ 与 exp_wind/, 文件名保持一致方便横向对比
        write_metrics_csv(all_metrics, out_dir + "/path_comparison.csv");
        if (!rrt_fluctuation.empty()) {
            write_rrt_fluctuation_csv(rrt_fluctuation, out_dir + "/rrt_fluctuation.csv");
        }
        if (wind_loaded) {
            write_wind_metrics_csv(all_metrics,
                                   out_dir + "/path_wind_metrics.csv",
                                   use_wind_cost);
        }
        std::cout << "\n[Done] 全部 " << num_pairs << " 组实验完成, 共 "
                  << all_metrics.size() << " 条记录, 产物目录: " << out_dir << "\n";

    // 释放内存
    if (dem.data) {
        delete[] dem.data;
        dem.data = nullptr;
    }

    return 0;
}

int main(int argc, char* argv[]) {

    // buildESDF();
    // buildESDFGradient();

    return runPathPlanningComparison(argc, argv);
}