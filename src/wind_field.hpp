#ifndef WIND_FIELD_HPP
#define WIND_FIELD_HPP

// ============================================================================
//  WindField — 三维插值风场查询类 (节能航线规划, 方案 B: 统一查询接口)
//
//  数据源: interpWind3D.py 生成的 interpolated_wind_YYYYMMDD_HHMM_2m.nc
//    - 坐标变量: lon/lat (WGS84 度, 1D) + height_agl (m, 1D, 升序)
//    - 数据变量: u, v (m/s), 维度 (height_agl, lat, lon)
//    - 全局属性 x/y_origin_7131 + grid_dx/dy_m 携带 EPSG:7131 网格锚点,
//      本类据此在规划坐标系 (EPSG:7131, 与 ESDF/DEM 一致) 中精确索引,
//      无需任何投影变换.
//
//  查询接口:
//    query(x, y, z_asl, dem) -> (u, v)
//      水平: (x,y) 在 7131 规则网格上直接算小数索引, 双线性;
//      垂直: h_agl = z_asl - DEM(x,y) (最近像元), 在 AGL 层间线性插值.
//    edgeCostCoeff(a, b, dem) -> 1 + λ_w · max(0, 顶风分量)
//      边中点风矢量在边水平方向上的投影; 顺风为负不计 (不奖励).
//      系数恒 ≥ 1, A* 欧氏启发式保持 admissible.
//
//  语义约定:
//    - 障碍体内/地下 (z ≤ DEM) 的风已由 nc 生成端置 0, 与 ESDF 障碍体
//      判定一致; 规划器在 ESDF≤0 处本就不会扩展, 无需重复判定.
//    - FMM 受 Eikonal 各向同性限制, 只能用风速模长 |w| (见 fmm_optimizer.hpp),
//      A*/RRT* 使用方向性顶风分量.
// ============================================================================

#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <gdal_priv.h>
#include "dem_data.hpp"

class WindField {
public:
    // 加载 nc 并解析网格; lambda_w: 顶风惩罚系数 λ_w (/ (m/s))
    bool load(const std::string& nc_path, double lambda_w = 0.05);

    bool   valid()  const { return valid_; }
    double lambda() const { return lambda_w_; }

    // (x,y) = EPSG:7131 平面坐标 (m), z_asl = 海拔 (m); 输出 (u,v) m/s
    void query(double x, double y, double z_asl, const DEMData& dem,
               double& u, double& v) const;

    // 边代价系数: 1 + λ_w·max(0, w·d̂_horiz)  (恒 ≥ 1)
    double edgeCostCoeff(double ax, double ay, double az,
                         double bx, double by, double bz,
                         const DEMData& dem) const;

private:
    double at(const std::vector<float>& a, int i, int j, int k) const {
        return a[((size_t)k * ny_ + j) * nx_ + i];
    }

    bool    valid_ = false;
    double  lambda_w_ = 0.05;                       // 顶风惩罚系数 (/m/s)
    int     nx_ = 0, ny_ = 0, nz_ = 0;              // 网格维度
    double  x0_ = 0.0, y0_ = 0.0;                   // 西南角网格中心 (7131)
    double  dx_ = 2.0, dy_ = 2.0;                   // 网格步长 (m)
    bool    h_uniform_ = true;                      // AGL 层是否均匀
    double  h0_ = 0.0, dh_ = 2.0;                   // AGL 层锚点 + 间隔
    std::vector<float> h_agl_;                      // [nz] 升序 (m)
    std::vector<float> u_, v_;                      // [k][j][i] 行优先 (j 自南向北)
};

// ----------------------------------------------------------------------------
// 加载: GDAL netCDF 子数据集 u/v, 逐 band (AGL 层) 读取
// ----------------------------------------------------------------------------
inline bool WindField::load(const std::string& nc_path, double lambda_w) {
    valid_ = false;
    lambda_w_ = lambda_w;

    const std::string sub_u = "netcdf:\"" + nc_path + "\":u";
    const std::string sub_v = "netcdf:\"" + nc_path + "\":v";
    GDALDataset* du = (GDALDataset*)GDALOpen(sub_u.c_str(), GA_ReadOnly);
    GDALDataset* dv = (GDALDataset*)GDALOpen(sub_v.c_str(), GA_ReadOnly);
    if (!du || !dv) {
        std::cerr << "[Wind] 无法打开风场子数据集: "
                  << (!du ? sub_u : sub_v) << "\n";
        if (du) GDALClose(du);
        if (dv) GDALClose(dv);
        return false;
    }

    nx_ = du->GetRasterXSize();
    ny_ = du->GetRasterYSize();
    nz_ = du->GetRasterCount();
    if (nx_ < 2 || ny_ < 2 || nz_ < 2) {
        std::cerr << "[Wind] 风场维度异常: " << nx_ << "x" << ny_
                  << " x" << nz_ << " 层\n";
        GDALClose(du); GDALClose(dv);
        return false;
    }

    // ---- 7131 网格锚点 (全局属性, 由 interpWind3D.py 写入) ----
    // 双保险: 不同 GDAL 版本对 netCDF 全局属性的暴露方式不同, 子数据集上
    // 可能带 "NC_GLOBAL#" 前缀 (GDAL netCDF 驱动惯例) 或为裸键;
    // 先查带前缀键, 查不到再回退裸键.
    auto get_global = [du](const char* name) -> const char* {
        if (const char* v = du->GetMetadataItem(
                (std::string("NC_GLOBAL#") + name).c_str()))
            return v;
        return du->GetMetadataItem(name);
    };
    const char* x0s = get_global("x_origin_7131");
    const char* y0s = get_global("y_origin_7131");
    const char* dxs = get_global("grid_dx_m");
    const char* dys = get_global("grid_dy_m");
    if (!x0s || !y0s || !dxs || !dys) {
        std::cerr << "[Wind] nc 缺少 7131 网格锚点属性 (x/y_origin_7131, "
                     "grid_dx/dy_m), 请用新版 interpWind3D.py 重新生成\n";
        GDALClose(du); GDALClose(dv);
        return false;
    }
    x0_ = std::atof(x0s);
    y0_ = std::atof(y0s);
    dx_ = std::atof(dxs);
    dy_ = std::atof(dys);

    // ---- 行序判定: lat 升序(南→北)时 GDAL geotransform dy>0 且行 0=南;
    //      仅在明确北朝上 (dy<0) 时才翻转读取; gt 缺失 (dy=0) 时按 nc 原始
    //      写入顺序 (lat 升序, 行 0=南) 处理 ----
    double gt[6] = {0, 0, 0, 0, 0, 0};
    du->GetGeoTransform(gt);
    const bool south_up = !(gt[5] < 0.0);

    // ---- AGL 层值 (band 元数据 NETCDF_DIM_height_agl) ----
    h_agl_.resize(nz_);
    for (int k = 0; k < nz_; ++k) {
        const char* hs =
            du->GetRasterBand(k + 1)->GetMetadataItem("NETCDF_DIM_height_agl");
        if (!hs) {
            std::cerr << "[Wind] band " << (k + 1)
                      << " 缺少 NETCDF_DIM_height_agl 元数据\n";
            GDALClose(du); GDALClose(dv);
            return false;
        }
        h_agl_[k] = (float)std::atof(hs);
    }
    h_uniform_ = true;
    dh_ = h_agl_[1] - h_agl_[0];
    for (int k = 2; k < nz_; ++k) {
        if (std::fabs((h_agl_[k] - h_agl_[k - 1]) - dh_) > 1e-4) {
            h_uniform_ = false;
            break;
        }
    }
    h0_ = h_agl_[0];

    // ---- 读取 u/v (逐层 RasterIO + 按行序翻转) ----
    const size_t n2 = (size_t)nx_ * ny_;
    u_.assign(n2 * nz_, 0.0f);
    v_.assign(n2 * nz_, 0.0f);
    std::vector<float> buf(n2);
    for (int k = 0; k < nz_; ++k) {
        for (int ivar = 0; ivar < 2; ++ivar) {
            GDALDataset* ds = ivar == 0 ? du : dv;
            std::vector<float>& dst = ivar == 0 ? u_ : v_;
            if (ds->GetRasterBand(k + 1)->RasterIO(
                    GF_Read, 0, 0, nx_, ny_, buf.data(), nx_, ny_,
                    GDT_Float32, 0, 0) != CE_None) {
                std::cerr << "[Wind] 读取 band " << (k + 1) << " 失败\n";
                GDALClose(du); GDALClose(dv);
                return false;
            }
            for (int j = 0; j < ny_; ++j) {
                const int src = south_up ? j : (ny_ - 1 - j);
                std::memcpy(&dst[((size_t)k * ny_ + j) * nx_],
                            &buf[(size_t)src * nx_], sizeof(float) * nx_);
            }
        }
    }
    GDALClose(du);
    GDALClose(dv);

    // ---- 摘要 + 风速极值抽样 (每 8 层抽 1) ----
    double wmin = 1e30, wmax = -1e30;
    for (int k = 0; k < nz_; k += std::max(1, nz_ / 8)) {
        for (int j = 0; j < ny_; j += std::max(1, ny_ / 8)) {
            for (int i = 0; i < nx_; i += std::max(1, nx_ / 8)) {
                double uu = at(u_, i, j, k), vv = at(v_, i, j, k);
                double w = std::sqrt(uu * uu + vv * vv);
                wmin = std::min(wmin, w);
                wmax = std::max(wmax, w);
            }
        }
    }

    valid_ = true;
    std::cout << "[Wind] 风场加载成功: " << nc_path << "\n"
              << "        网格 " << nx_ << "x" << ny_ << " x" << nz_
              << " (dx=" << dx_ << " m), 7131 范围 x[" << x0_ << ", "
              << x0_ + (nx_ - 1) * dx_ << "] y[" << y0_ << ", "
              << y0_ + (ny_ - 1) * dy_ << "]\n"
              << "        AGL " << h_agl_.front() << "~" << h_agl_.back()
              << " m (" << nz_ << " 层, "
              << (h_uniform_ ? "均匀" : "非均匀") << "), "
              << "抽样风速 " << wmin << "~" << wmax << " m/s, "
              << "λ_w=" << lambda_w_ << "\n";
    return true;
}

// ----------------------------------------------------------------------------
// 查询: 水平 7131 直索引双线性 + 垂直 AGL 线性
// ----------------------------------------------------------------------------
inline void WindField::query(double x, double y, double z_asl,
                             const DEMData& dem,
                             double& u, double& v) const {
    if (!valid_) {
        u = v = 0.0;
        return;
    }

    // 水平小数索引 (规划范围 ⊆ 风场网格, 越界仅防御性钳制)
    double fi = (x - x0_) / dx_;
    double fj = (y - y0_) / dy_;
    fi = std::clamp(fi, 0.0, (double)(nx_ - 1));
    fj = std::clamp(fj, 0.0, (double)(ny_ - 1));

    // 垂直: 海拔 → AGL (最近像元 DEM, 与 ESDF/参考高程同源)
    const double h_agl = z_asl - dem.getElevation(x, y);
    double fk;
    if (h_uniform_) {
        fk = (h_agl - h0_) / dh_;
    } else {
        // 二分找最大 k: h_agl_[k] <= h_agl, 再层内线性
        int lo = 0, hi = nz_ - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (h_agl_[mid] <= h_agl) lo = mid;
            else                      hi = mid - 1;
        }
        fk = (double)lo;
        if (lo < nz_ - 1) {
            const double span = h_agl_[lo + 1] - h_agl_[lo];
            if (span > 1e-6)
                fk += std::clamp((h_agl - h_agl_[lo]) / span, 0.0, 1.0);
        }
    }
    fk = std::clamp(fk, 0.0, (double)(nz_ - 1));

    // 三线性插值 (x → y → z)
    const int i0 = (int)fi, j0 = (int)fj, k0 = (int)fk;
    const int i1 = std::min(i0 + 1, nx_ - 1);
    const int j1 = std::min(j0 + 1, ny_ - 1);
    const int k1 = std::min(k0 + 1, nz_ - 1);
    const double ti = fi - i0, tj = fj - j0, tk = fk - k0;

    auto L = [](double a, double b, double t) { return a + (b - a) * t; };
    u = L(L(L(at(u_, i0, j0, k0), at(u_, i1, j0, k0), ti),
            L(at(u_, i0, j1, k0), at(u_, i1, j1, k0), ti), tj),
          L(L(at(u_, i0, j0, k1), at(u_, i1, j0, k1), ti),
            L(at(u_, i0, j1, k1), at(u_, i1, j1, k1), ti), tj), tk);
    v = L(L(L(at(v_, i0, j0, k0), at(v_, i1, j0, k0), ti),
            L(at(v_, i0, j1, k0), at(v_, i1, j1, k0), ti), tj),
          L(L(at(v_, i0, j0, k1), at(v_, i1, j0, k1), ti),
            L(at(v_, i0, j1, k1), at(v_, i1, j1, k1), ti), tj), tk);
}

// ----------------------------------------------------------------------------
// 边代价系数: 1 + λ_w · max(0, 顶风分量)
//   顶风分量 = 边中点风矢量 · 边水平单位方向 (负值=顺风, 不奖励)
//   纯垂直边无水平方向 → 系数 1 (简化模型不考虑悬停/垂直风载)
// ----------------------------------------------------------------------------
inline double WindField::edgeCostCoeff(double ax, double ay, double az,
                                       double bx, double by, double bz,
                                       const DEMData& dem) const {
    if (!valid_) return 1.0;
    const double dhx = bx - ax, dhy = by - ay;
    const double dh = std::sqrt(dhx * dhx + dhy * dhy);
    if (dh < 1e-9) return 1.0;

    double wu, wv;
    query(0.5 * (ax + bx), 0.5 * (ay + by), 0.5 * (az + bz), dem, wu, wv);
    const double headwind = (wu * dhx + wv * dhy) / dh;
    return 1.0 + lambda_w_ * std::max(0.0, headwind);
}

#endif // WIND_FIELD_HPP
