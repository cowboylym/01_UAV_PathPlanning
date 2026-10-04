#!/usr/bin/env python
"""
VTK 体渲染 - 读取 .mhd 文件并进行三维体绘制
基于 VTK 官方示例 MedicalDemo4
"""

import sys
import vtk
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOImage import vtkMetaImageReader
from vtkmodules.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
# 确保 OpenGL2 和 Volume 后端被加载
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingVolumeOpenGL2


def main():
    # 检查命令行参数
    # if len(sys.argv) < 2:
    #     print("用法: python volume_render.py <文件路径.mhd>")
    #     print("示例: python volume_render.py ./workspace/build/uav_terrain_3d_sdf.mhd")
    #     sys.exit(1)

    file_name = "./workspace/build/uav_terrain_3d_sdf.mhd"

    # ---- 1. 创建渲染器、渲染窗口和交互器 ----
    colors = vtkNamedColors()
    colors.SetColor("BkgColor", [51, 77, 102, 255])  # 深蓝色背景

    renderer = vtkRenderer()
    renderer.SetBackground(colors.GetColor3d("BkgColor"))

    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)

    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # ---- 2. 读取 .mhd 文件 ----
    reader = vtkMetaImageReader()
    reader.SetFileName(file_name)
    reader.Update()  # 执行读取

    # 打印数据信息（便于调试）
    image_data = reader.GetOutput()
    if image_data:
        dims = image_data.GetDimensions()
        print(f"数据维度: {dims[0]} x {dims[1]} x {dims[2]}")
        scalar_range = image_data.GetScalarRange()
        print(f"标量范围: [{scalar_range[0]:.2f}, {scalar_range[1]:.2f}]")

    # ---- 3. 创建体渲染映射器 ----
    # 使用固定点光线投射映射器，稳定性好，兼容性强
    volume_mapper = vtkFixedPointVolumeRayCastMapper()
    volume_mapper.SetInputConnection(reader.GetOutputPort())
    volume_mapper.SetBlendModeToComposite()  # 合成模式（alpha compositing）

        # ---- 4. 定义颜色传递函数 ----
    volume_color = vtkColorTransferFunction()
    scalar_range = image_data.GetScalarRange()
    min_val = scalar_range[0]
    max_val = scalar_range[1]
    
    if min_val < -100 or max_val > 500:
        # SDF 分支：≤0 红色，>0 从红渐变到绿
        volume_color.AddRGBPoint(0.0, 1.0, 0.0, 0.3)    # 红
        volume_color.AddRGBPoint(max_val*0.5, 1.0, 1.0, 0.3)  # 黄（红+绿）
        volume_color.AddRGBPoint(max_val, 0.0, 1.0, 0.3)      # 绿
    else:
        # CT 分支保持不变
        volume_color.AddRGBPoint(0, 0.0, 0.0, 0.0)
        volume_color.AddRGBPoint(500, 1.0, 0.5, 0.3)
        volume_color.AddRGBPoint(1000, 1.0, 0.5, 0.3)
        volume_color.AddRGBPoint(1150, 1.0, 1.0, 0.9)

    # ---- 5. 定义不透明度传递函数 ----
    # 控制不同体素值的透明程度
    volume_scalar_opacity = vtk.vtkPiecewiseFunction()
    
    if min_val < -100 or max_val > 500:
        # SDF 或通用数据：在 0 表面附近高不透明度，远离则透明
        volume_scalar_opacity.AddPoint(min_val, 0.0)
        volume_scalar_opacity.AddPoint(-10, 0.0)
        volume_scalar_opacity.AddPoint(0, 0.8)      # 表面处高不透明度
        volume_scalar_opacity.AddPoint(10, 0.2)
        volume_scalar_opacity.AddPoint(max_val, 0.0)
    else:
        # CT 医学数据风格
        volume_scalar_opacity.AddPoint(0, 0.00)
        volume_scalar_opacity.AddPoint(500, 0.15)
        volume_scalar_opacity.AddPoint(1000, 0.15)
        volume_scalar_opacity.AddPoint(1150, 0.85)

    # ---- 6. 定义梯度不透明度函数（可选，增强边界清晰度） ----
    # 在平坦区域降低不透明度，在边界处保持高不透明度
    volume_gradient_opacity = vtk.vtkPiecewiseFunction()
    volume_gradient_opacity.AddPoint(0, 0.0)
    volume_gradient_opacity.AddPoint(90, 0.5)
    volume_gradient_opacity.AddPoint(100, 1.0)

    # ---- 7. 组装体属性 ----
    volume_property = vtkVolumeProperty()
    volume_property.SetColor(volume_color)
    volume_property.SetScalarOpacity(volume_scalar_opacity)
    volume_property.SetGradientOpacity(volume_gradient_opacity)
    volume_property.SetInterpolationTypeToLinear()  # 线性插值，质量更高
    volume_property.ShadeOn()  # 开启光照，增强立体感
    volume_property.SetAmbient(0.4)
    volume_property.SetDiffuse(0.6)
    volume_property.SetSpecular(0.2)

    # ---- 8. 创建体对象并添加到场景 ----
    volume = vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    renderer.AddVolume(volume)

    # ---- 9. 设置相机视角 ----
    renderer.GetActiveCamera().Azimuth(45)   # 旋转45度
    renderer.GetActiveCamera().Elevation(30) # 俯仰30度
    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()

    # ---- 10. 开始渲染与交互 ----
    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()