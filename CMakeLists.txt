cmake_minimum_required(VERSION 3.10)
project(UavSdfGeneratorProject)

# std::filesystem 与 std::clamp 需要 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================================
# 1. 配置 1：寻找系统的 GDAL 库
# =========================================================================
find_path(GDAL_INCLUDE_DIR gdal_priv.h PATHS /usr/include/gdal)
find_library(GDAL_LIBRARY gdal)

if(GDAL_INCLUDE_DIR AND GDAL_LIBRARY)
    message(STATUS "Found system GDAL: ${GDAL_LIBRARY}")
else()
    message(FATAL_ERROR "System GDAL not found")
endif()

# =========================================================================
# 2. 配置 2：寻找系统的 ITK 库
# =========================================================================
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# =========================================================================
# 3. 配置 3：寻找系统的 OMPL 库 🌟 (新增)
# =========================================================================
find_package(ompl REQUIRED)
if(OMPL_FOUND)
    message(STATUS "Found OMPL version: ${OMPL_VERSION}")
else()
    message(FATAL_ERROR "OMPL not found")
endif()

# 告诉 ITK 避开与系统 Eigen 的宏命名冲突
add_definitions(-DITK_USE_SYSTEM_EIGEN=ON) 
add_definitions(-DEIGEN_MPL2_ONLY)

# =========================================================================
# 4. 目标配置（Target Configuration）
# =========================================================================
add_executable(uav_planner 
    src/main.cpp
)

# 包含头文件目录：同时包含 GDAL、ITK 和 OMPL 的头文件
target_include_directories(uav_planner PRIVATE 
    ${GDAL_INCLUDE_DIR}
    ${ITK_INCLUDE_DIRS}
    ${OMPL_INCLUDE_DIRS} # 🌟 显式引入 OMPL 头文件包含路径
)

# 链接库文件：同时链接 GDAL、ITK 以及 OMPL 的核心库
target_link_libraries(uav_planner PRIVATE 
    ${GDAL_LIBRARY}
    ${ITK_LIBRARIES}
    ${OMPL_LIBRARIES}    # 🌟 显式链接 OMPL 动态库
)