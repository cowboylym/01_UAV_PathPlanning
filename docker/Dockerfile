# ==========================================
# 阶段 1：编译环境 (Builder)
# ==========================================
FROM ubuntu:22.04 AS builder

# 1. apt 换源（清华源，可根据需要替换）
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 2. 安装基础编译工具链包 (Git, CMake, GCC/Clang, 压缩工具等)
RUN apt-get update && apt-get install -y \
    build-essential cmake git curl zip unzip tar pkg-config \
    libgdal-dev gdal-bin \
    gdb gdbserver \
    libxt-dev \
    python3 \
    python3-pip \
    python3-dev \
    # 关键：安装用于 PyVista 3D 离屏/虚拟渲染的图形底座 
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    xvfb \
    libegl1-mesa \
    libboost-all-dev \
    libosmesa6 \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*
# 3. 获取并初始化 vcpkg（使用 Gitee 镜像加速）
# WORKDIR /opt
# RUN git clone https://github.com/Microsoft/vcpkg && \
#     ./vcpkg/bootstrap-vcpkg.sh

# # 4. （可选）设置 vcpkg 环境变量，方便后续使用
# ENV VCPKG_ROOT=/opt/vcpkg
# ENV PATH="/opt/vcpkg:${PATH}"

# 2. 下载并编译安装 ITK 库（由于ITK较大，编译需要一些时间）
WORKDIR /tmp
RUN git clone --depth 1 --branch release-5.4 https://gitee.com/cowboylym/ITK.git \
    && mkdir ITK-build \
    && cd ITK-build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF ../ITK \
    && make -j$(nproc) \
    && make install \
    && cd / \
    && rm -rf /tmp/ITK /tmp/ITK-build


    # 下载 OMPL 源码并编译 (以 1.6.0 版本为例，可按需修改)
WORKDIR /tmp
# 使用 git 克隆 OMPL 官方仓库
# --depth 1 表示只克隆单层提交历史，能大幅加快克隆速度
# --branch 1.6.0 指定安装 1.6.0 稳定版（可以根据需要改为 1.5.2 或 main）
WORKDIR /tmp
RUN git clone --depth 1 --branch 2.0.1 https://gitee.com/cowboylym/ompl.git \
    && cd ompl \
    && mkdir build && cd build \
    # -DOMPL_BUILD_PYBINDINGS=OFF 显式关闭 Python 绑定，专为纯 C++ 环境加速
    && cmake -DCMAKE_BUILD_TYPE=Release -DOMPL_BUILD_PYBINDINGS=OFF .. \
    && make -j$(nproc) \
    && make install

# 3. 安装 Python 依赖库
# 升级 pip 并安装 numpy, simpleitk, pyvista 
RUN pip3 install --no-cache-dir --upgrade pip --index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy rasterio matplotlib pyvista SimpleITK
# 4. 设置 PyVista 的环境变量，使其在 Docker 容器内优先采用内建的虚拟窗口和软渲染
ENV PYVISTA_OFF_SCREEN=false
ENV PYVISTA_USE_IPYVTK=false
