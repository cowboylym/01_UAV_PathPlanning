// dem_data.hpp
#pragma once
#include <gdal_priv.h>
#include <vector>
#include <cmath>
#include <algorithm>

struct DEMData {
    int width, height;
    float* data;                 // 高程数组（行优先），所有权由外部管理，注意析构时不删除（可选用 shared_ptr）
    double geoTransform[6];
    double nodata;
    bool hasNodata;
    double min_elevation;        // 新增：有效高程最小值
    double max_elevation;        // 新增：有效高程最大值

    // 构造函数：初始化指针为空，极值置为默认
    DEMData() : data(nullptr), width(0), height(0), hasNodata(false), 
                nodata(-9999.0), min_elevation(1e30), max_elevation(-1e30) {
        std::fill(geoTransform, geoTransform+6, 0.0);
    }

    // 如需深拷贝，需重写拷贝构造/赋值，此处仅作示意，实际使用时注意 data 的内存管理
    ~DEMData() {
        // 注意：data 若由外部 new[] 分配，且此处拥有所有权则 delete[]，否则置空。
        // 建议在构造时明确所有权，或使用 std::shared_ptr<float>。
        // 这里为安全，不自动释放，由调用者管理。
    }

    // 物理坐标转像素坐标
    bool worldToPixel(double wx, double wy, int& col, int& row) const {
        double inv = 1.0 / (geoTransform[1] * geoTransform[5] - geoTransform[2] * geoTransform[4]);
        double col_d = (geoTransform[5] * (wx - geoTransform[0]) - geoTransform[2] * (wy - geoTransform[3])) * inv;
        double row_d = (-geoTransform[4] * (wx - geoTransform[0]) + geoTransform[1] * (wy - geoTransform[3])) * inv;
        col = (int)std::floor(col_d + 0.5);
        row = (int)std::floor(row_d + 0.5);
        return (col >= 0 && col < width && row >= 0 && row < height);
    }

    // 双线性插值获取高程（可选用，此处保留最近邻）
    double getElevation(double wx, double wy) const {
        int col, row;
        if (!worldToPixel(wx, wy, col, row)) return 0.0;
        int c = std::max(0, std::min(width-1, col));
        int r = std::max(0, std::min(height-1, row));
        float val = data[r * width + c];
        if (hasNodata && std::abs(val - nodata) < 1e-3) return 0.0;
        return val;
    }
};