// 简单的基于DEM高度图的射线碰撞检测
bool is_collision_raycasting(double x1, double y1, double z1, 
                             double x2, double y2, double z2, 
                             const cv::Mat& dem_map) { // 假设使用OpenCV读取DEM
    double dx = std::abs(x2 - x1);
    double dy = std::abs(y2 - y1);
    int x = int(x1), y = int(y1);
    int n = 1 + int(dx) + int(dy);
    int x_inc = (x2 > x1) ? 1 : -1;
    int y_inc = (y2 > y1) ? 1 : -1;
    double error = dx - dy;
    dx *= 2; dy *= 2;

    for (; n > 0; --n) {
        if (x >= 0 && x < dem_map.cols && y >= 0 && y < dem_map.rows) {
            float z_dem = dem_map.at<float>(y, x); // 获取DEM高度
            // 计算当前射线高度 z(t)
            double t = (x - x1) / (x2 - x1 + 1e-9); // 简化的插值系数
            double z_curr = z1 + t * (z2 - z1);
            if (z_curr <= z_dem) return true; // 发生碰撞
        }
        if (error > 0) { x += x_inc; error -= dy; }
        else { y += y_inc; error += dx; }
    }
    return false;
}