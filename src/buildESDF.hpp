//main.cpp
#pragma once
#include <iostream>
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>
#include <limits>
#include <string>

// GDAL 用于读取 GeoTIFF
#include <gdal_priv.h>

// ITK
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkVectorImage.h>

#include "rrt.hpp"

// -----------------------------------------------------------------------------
// buildESDF
//   从 DEM GeoTIFF 构建 3D 体素掩膜, 计算 ESDF (欧氏有向距离场), 保存到磁盘.
//   输出: /workspace/01_uav_path_planning/data/SF_Downtown_sdf.mhd
// -----------------------------------------------------------------------------
int buildESDF() {
    // -------------------------------------------------------------------------
    // 1. 无人机分辨率与安全冗余配置
    // -------------------------------------------------------------------------
    const float voxel_size_xy = 1.0f;          // 水平体素分辨率：1米一格
    const float voxel_size_z  = 1.0f;          // 高度体素分辨率：1米一格
    const float max_fly_height_buffer = 50.0f; // 在最高山顶之上的安全空中规划冗余（米）

    //数值大于 0（正值：D > 0）—— 安全空域
    //数值等于 0（零值：D = 0）—— 地形表面
    //数值小于 0（负值：D < 0）—— 障碍物内部

    // -------------------------------------------------------------------------
    // 2. 使用 GDAL 读取 2D DEM (GeoTIFF)
    // -------------------------------------------------------------------------
    GDALAllRegister();
    GDALDataset* poDS = (GDALDataset*)GDALOpen("/workspace/01_uav_path_planning/data/SF_Downtown.tif", GA_ReadOnly);
    if (!poDS) {
        std::cerr << "错误：无法打开 DEM 文件！" << std::endl;
        return 1;
    }

    int width = poDS->GetRasterXSize();
    int height = poDS->GetRasterYSize();

    // 获取 2D DEM 的地理仿射变换参数
    double adfGeoTransform[6];
    poDS->GetGeoTransform(adfGeoTransform);
    double origin_x = adfGeoTransform[0]; // 左上角 X 物理坐标
    double origin_y = adfGeoTransform[3]; // 左上角 Y 物理坐标
    double pixel_w  = adfGeoTransform[1]; // 像素宽度（米）
    double pixel_h  = adfGeoTransform[5]; // 像素高度（米，通常为负数）

    // 计算实际地理范围的四至
    double min_geo_x = origin_x;
    double max_geo_x = origin_x + width * pixel_w;
    double max_geo_y = origin_y;
    double min_geo_y = origin_y + height * pixel_h; // pixel_h 为负数

    // 读取高程 data 到内存
    float* dem_data = new float[width * height];
    poDS->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, width, height, dem_data, width, height, GDT_Float32, 0, 0);

    // 获取 NoData 值
    int has_nodata = 0;
    float nodata_val = poDS->GetRasterBand(1)->GetNoDataValue(&has_nodata);

    std::cout << "DEM 加载成功 | 尺寸: " << width << " x " << height << std::endl;
    std::cout << "DEM 物理分辨率: W=" << std::abs(pixel_w) << "m, H=" << std::abs(pixel_h) << "m" << std::endl;
    std::cout << "DEM 地理范围 X: [" << min_geo_x << ", " << max_geo_x << "]" << std::endl;
    std::cout << "DEM 地理范围 Y: [" << min_geo_y << ", " << max_geo_y << "]" << std::endl;

    // -------------------------------------------------------------------------
    // 3. 动态计算 DEM 的最大和最小高程
    // -------------------------------------------------------------------------
    float min_dem_z = std::numeric_limits<float>::max();
    float max_dem_z = std::numeric_limits<float>::lowest();
    size_t valid_pixel_count = 0;

    for (int i = 0; i < width * height; ++i) {
        float val = dem_data[i];
        if (has_nodata && std::abs(val - nodata_val) < 1e-3) continue;
        if (val < min_dem_z) min_dem_z = val;
        if (val > max_dem_z) max_dem_z = val;
        valid_pixel_count++;
    }

    if (valid_pixel_count == 0) {
        std::cerr << "错误：DEM 文件中未检测到有效的地形高程数据！" << std::endl;
        delete[] dem_data;
        GDALClose(poDS);
        return 1;
    }

    float min_fly_height = std::floor(min_dem_z);
    float max_fly_height = std::ceil(max_dem_z + max_fly_height_buffer);

    std::cout << "--------------------------------------------------------\n";
    std::cout << "地形动态统计结果:\n";
    std::cout << " -> 地形最低高程: " << min_dem_z << " 米\n";
    std::cout << " -> 地形最高高程: " << max_dem_z << " 米\n";
    std::cout << " -> 3D 规划底面 (min_fly_height): " << min_fly_height << " 米\n";
    std::cout << " -> 3D 规划顶面 (max_fly_height): " << max_fly_height << " 米\n";
    std::cout << "--------------------------------------------------------\n";

    // -------------------------------------------------------------------------
    // 4. 构建并初始化 ITK 3D 体素掩膜
    // -------------------------------------------------------------------------
    auto mask3D = Mask3DImageType::New();

    Mask3DImageType::SizeType size3D;
    size3D[0] = std::ceil((max_geo_x - min_geo_x) / voxel_size_xy);
    size3D[1] = std::ceil((max_geo_y - min_geo_y) / voxel_size_xy);
    size3D[2] = std::ceil((max_fly_height - min_fly_height) / voxel_size_z);

    Mask3DImageType::RegionType region3D;
    region3D.SetSize(size3D);
    mask3D->SetRegions(region3D);

    Mask3DImageType::SpacingType spacing3D;
    spacing3D[0] = voxel_size_xy;
    spacing3D[1] = voxel_size_xy;
    spacing3D[2] = voxel_size_z;
    mask3D->SetSpacing(spacing3D);

    Mask3DImageType::PointType origin3D;
    origin3D[0] = min_geo_x;
    origin3D[1] = min_geo_y;
    origin3D[2] = min_fly_height;
    mask3D->SetOrigin(origin3D);

    mask3D->Allocate();
    mask3D->FillBuffer(0);

    std::cout << "3D 规划体素场网格尺寸: " << size3D[0] << " x " << size3D[1] << " x " << size3D[2] << std::endl;

    // -------------------------------------------------------------------------
    // 5. 将 2D DEM 数据投影到 3D 掩膜
    // -------------------------------------------------------------------------
    std::cout << "正在将 DEM 投影至 3D 栅格..." << std::endl;

    for (unsigned int z = 0; z < size3D[2]; ++z) {
        double current_z = min_fly_height + (z * voxel_size_z);
        for (unsigned int y = 0; y < size3D[1]; ++y) {
            double phys_y = min_geo_y + (y * voxel_size_xy);
            for (unsigned int x = 0; x < size3D[0]; ++x) {
                double phys_x = min_geo_x + (x * voxel_size_xy);

                int dem_col = std::floor((phys_x - origin_x) / pixel_w);
                int dem_row = std::floor((max_geo_y - phys_y) / std::abs(pixel_h));

                if (dem_col >= 0 && dem_col < width && dem_row >= 0 && dem_row < height) {
                    float dem_elevation = dem_data[dem_row * width + dem_col];
                    if ((!has_nodata || std::abs(dem_elevation - nodata_val) > 1e-3) && current_z <= dem_elevation) {
                        Mask3DImageType::IndexType idx3D;
                        idx3D[0] = x; idx3D[1] = y; idx3D[2] = z;
                        mask3D->SetPixel(idx3D, 1);
                    }
                }
            }
        }
    }

    delete[] dem_data;
    GDALClose(poDS);

    // -------------------------------------------------------------------------
    // 6. ITK SignedMaurerDistanceMapImageFilter 计算 3D ESDF
    // -------------------------------------------------------------------------
    std::cout << "开始利用 ITK Maurer 滤波器并行计算 3D 欧氏距离场..." << std::endl;

    auto distanceFilter = itk::SignedMaurerDistanceMapImageFilter<Mask3DImageType, SDFImageType>::New();
    distanceFilter->SetInput(mask3D);
    distanceFilter->SetUseImageSpacing(true);
    distanceFilter->SetInsideIsPositive(false);
    distanceFilter->SetNumberOfWorkUnits(std::thread::hardware_concurrency());

    try {
        distanceFilter->Update();
    } catch (const itk::ExceptionObject& e) {
        std::cerr << "致命错误：ITK 距离变换计算失败！\n" << e.GetDescription() << std::endl;
        return 1;
    }

    std::cout << "3D 欧氏有向距离场 (ESDF) 计算成功完成！" << std::endl;

    // -------------------------------------------------------------------------
    // 7. 保存 SDF
    // -------------------------------------------------------------------------
    SDFImageType::Pointer sdfImage = distanceFilter->GetOutput();

    auto writer = itk::ImageFileWriter<SDFImageType>::New();
    writer->SetFileName("/workspace/01_uav_path_planning/data/SF_Downtown_sdf.mhd");
    writer->SetInput(sdfImage);
    writer->Update();
    std::cout << "3D SDF 场已成功保存至: SF_Downtown_sdf.mhd" << std::endl;

    return 0;
}

// -----------------------------------------------------------------------------
// buildESDFGradient
//   从已保存的 ESDF MHD 文件读取标量距离场, 用中心差分计算梯度矢量场,
//   保存为矢量 MHD (3通道 gx,gy,gz).
//   为节省资源 (论文示意图无需全分辨率), 采用降采样: 每隔 shrink 个体素
//   取一个采样点, 在原网格上以 shrink 为步长做中心差分, 输出维度缩小
//   shrink 倍, spacing 放大 shrink 倍.
//   参数:
//     sdf_path  - 输入 ESDF 文件路径 (默认 SF_Downtown_sdf.mhd)
//     out_dir   - 输出目录 (默认 /workspace/01_uav_path_planning/data/)
//     shrink    - 降采样因子 (默认 5, 即每 5 个体素取 1 个, 计算量减少 125 倍)
//   输出:
//     {base}_grad.mhd - ITK VectorImage, 3 通道 float (gx, gy, gz)
// -----------------------------------------------------------------------------
int buildESDFGradient(const std::string& sdf_path = "/workspace/01_uav_path_planning/data/SF_Downtown_sdf.mhd",
                      const std::string& out_dir  = "/workspace/01_uav_path_planning/data/",
                      int shrink = 5)
{
    using GradVecImageType = itk::VectorImage<float, 3>;

    if (shrink < 1) shrink = 1;

    // 从 base name 推导输出文件名
    // e.g. "/workspace/01_uav_path_planning/data/SF_Downtown_sdf.mhd" -> "SF_Downtown"
    std::string base = sdf_path;
    {
        size_t slash = base.find_last_of("/\\");
        if (slash != std::string::npos) base = base.substr(slash + 1);
        size_t dot = base.rfind("_sdf.mhd");
        if (dot != std::string::npos) base = base.substr(0, dot);
        else {
            dot = base.rfind('.');
            if (dot != std::string::npos) base = base.substr(0, dot);
        }
    }
    std::string grad_path = out_dir + base + "_grad.mhd";

    std::cout << "========================================================\n";
    std::cout << "ESDF 梯度场计算 (降采样 shrink=" << shrink << ")\n";
    std::cout << "  输入 SDF : " << sdf_path << "\n";
    std::cout << "  输出梯度 : " << grad_path << "\n";
    std::cout << "========================================================\n";

    // 1. 读取 SDF 标量场
    auto reader = itk::ImageFileReader<SDFImageType>::New();
    reader->SetFileName(sdf_path);
    try {
        reader->Update();
    } catch (const itk::ExceptionObject& e) {
        std::cerr << "错误：无法读取 SDF 文件 " << sdf_path << "\n" << e.GetDescription() << std::endl;
        return 1;
    }
    SDFImageType::Pointer sdfImage = reader->GetOutput();

    SDFImageType::RegionType region3D   = sdfImage->GetLargestPossibleRegion();
    SDFImageType::SizeType   sdfSize    = region3D.GetSize();
    SDFImageType::SpacingType spacing3D = sdfImage->GetSpacing();
    SDFImageType::PointType   origin3D  = sdfImage->GetOrigin();

    const int nx = static_cast<int>(sdfSize[0]);
    const int ny = static_cast<int>(sdfSize[1]);
    const int nz = static_cast<int>(sdfSize[2]);
    const float sp = static_cast<float>(spacing3D[0]); // 原始 spacing (1.0m)

    // 降采样后的输出网格尺寸
    const int nx_s = std::max(2, (nx + shrink - 1) / shrink);
    const int ny_s = std::max(2, (ny + shrink - 1) / shrink);
    const int nz_s = std::max(2, (nz + shrink - 1) / shrink);
    const float sp_s = sp * shrink;  // 降采样后的物理步长

    std::cout << "SDF 原始网格  : " << nx << " x " << ny << " x " << nz
              << "  spacing=" << sp << "m\n";
    std::cout << "降采样后网格  : " << nx_s << " x " << ny_s << " x " << nz_s
              << "  spacing=" << sp_s << "m"
              << "  (计算量减少 " << (shrink*shrink*shrink) << " 倍)\n";
    std::cout << "正在计算梯度场 (中心差分, 步长=" << shrink << " 体素)..." << std::endl;

    // 2. 分配降采样后的输出图像
    GradVecImageType::SizeType  outSize;
    outSize[0] = nx_s; outSize[1] = ny_s; outSize[2] = nz_s;
    GradVecImageType::RegionType outRegion;
    outRegion.SetSize(outSize);
    GradVecImageType::SpacingType outSpacing;
    outSpacing[0] = outSpacing[1] = outSpacing[2] = sp_s;
    GradVecImageType::PointType outOrigin = origin3D;  // 原点保持一致

    auto gradVecImage = GradVecImageType::New();
    gradVecImage->SetRegions(outRegion);
    gradVecImage->SetSpacing(outSpacing);
    gradVecImage->SetOrigin(outOrigin);
    gradVecImage->SetNumberOfComponentsPerPixel(3);
    gradVecImage->Allocate();

    // 安全取值 lambda: 越界返回当前体素值 (边界退化为单边差分, 0梯度)
    auto getSDF = [&](int ix, int iy, int iz, float fallback) -> float {
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) return fallback;
        SDFImageType::IndexType idx;
        idx[0] = ix; idx[1] = iy; idx[2] = iz;
        return sdfImage->GetPixel(idx);
    };

    float max_gmag = 0.0f;
    float sum_gmag = 0.0f;
    size_t count_gmag = 0;

    // 3. 遍历降采样网格, 在原网格上以 shrink 为步长做中心差分
    for (int iz_s = 0; iz_s < nz_s; ++iz_s) {
        int iz = iz_s * shrink;              // 映射到原网格索引
        for (int iy_s = 0; iy_s < ny_s; ++iy_s) {
            int iy = iy_s * shrink;
            for (int ix_s = 0; ix_s < nx_s; ++ix_s) {
                int ix = ix_s * shrink;

                SDFImageType::IndexType idx;
                idx[0] = ix; idx[1] = iy; idx[2] = iz;
                float dc = sdfImage->GetPixel(idx);

                float gx, gy, gz;
                int ix_p = ix + shrink, ix_m = ix - shrink;
                int iy_p = iy + shrink, iy_m = iy - shrink;
                int iz_p = iz + shrink, iz_m = iz - shrink;

                // X 方向: 中心差分 (内部) / 单边差分 (边界)
                if (ix_m < 0) {
                    gx = (getSDF(ix_p, iy, iz, dc) - dc) / sp_s;
                } else if (ix_p >= nx) {
                    gx = (dc - getSDF(ix_m, iy, iz, dc)) / sp_s;
                } else {
                    gx = (getSDF(ix_p, iy, iz, dc) - getSDF(ix_m, iy, iz, dc)) / (2.0f * sp_s);
                }

                // Y 方向
                if (iy_m < 0) {
                    gy = (getSDF(ix, iy_p, iz, dc) - dc) / sp_s;
                } else if (iy_p >= ny) {
                    gy = (dc - getSDF(ix, iy_m, iz, dc)) / sp_s;
                } else {
                    gy = (getSDF(ix, iy_p, iz, dc) - getSDF(ix, iy_m, iz, dc)) / (2.0f * sp_s);
                }

                // Z 方向
                if (iz_m < 0) {
                    gz = (getSDF(ix, iy, iz_p, dc) - dc) / sp_s;
                } else if (iz_p >= nz) {
                    gz = (dc - getSDF(ix, iy, iz_m, dc)) / sp_s;
                } else {
                    gz = (getSDF(ix, iy, iz_p, dc) - getSDF(ix, iy, iz_m, dc)) / (2.0f * sp_s);
                }

                float gmag = std::sqrt(gx*gx + gy*gy + gz*gz);
                if (gmag > max_gmag) max_gmag = gmag;
                sum_gmag += gmag;
                count_gmag++;

                // 写入降采样后的矢量图像
                GradVecImageType::IndexType outIdx;
                outIdx[0] = ix_s; outIdx[1] = iy_s; outIdx[2] = iz_s;
                itk::VariableLengthVector<float> gvec(3);
                gvec[0] = gx;
                gvec[1] = gy;
                gvec[2] = gz;
                gradVecImage->SetPixel(outIdx, gvec);
            }
        }
    }

    float avg_gmag = (count_gmag > 0) ? (sum_gmag / static_cast<float>(count_gmag)) : 0.0f;
    std::cout << "梯度场计算完成:\n";
    std::cout << "  - 体素总数     : " << count_gmag << "\n";
    std::cout << "  - 最大|∇ESDF|  : " << max_gmag << " (理想值≈1.0)\n";
    std::cout << "  - 平均|∇ESDF|  : " << avg_gmag << "\n";

    // 4. 保存矢量梯度场 (3 通道: gx, gy, gz)
    auto gvecWriter = itk::ImageFileWriter<GradVecImageType>::New();
    gvecWriter->SetFileName(grad_path);
    gvecWriter->SetInput(gradVecImage);
    gvecWriter->Update();
    std::cout << "  - 矢量梯度场已保存: " << grad_path << " (3通道: gx,gy,gz)\n";
    std::cout << "========================================================\n";

    return 0;
}



// 从已打开的 GDALDataset 和已计算好的极值构建 DEMData
bool fillDEMDataFromGDAL(GDALDataset* poDS, float* dem_data, int width, int height,
                         double min_dem_z, double max_dem_z,
                         DEMData& dem)
{
    if (!poDS || !dem_data || width <= 0 || height <= 0) return false;

    dem.width = width;
    dem.height = height;
    dem.data = dem_data;
    dem.min_elevation = min_dem_z;
    dem.max_elevation = max_dem_z;

    double adfGeoTransform[6];
    poDS->GetGeoTransform(adfGeoTransform);
    for (int i = 0; i < 6; ++i) dem.geoTransform[i] = adfGeoTransform[i];

    int has_nodata = 0;
    double nodata_val = poDS->GetRasterBand(1)->GetNoDataValue(&has_nodata);
    dem.hasNodata = (has_nodata != 0);
    dem.nodata = nodata_val;

    return true;
}
