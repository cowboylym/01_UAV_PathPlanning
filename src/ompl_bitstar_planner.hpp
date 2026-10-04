#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

// OMPL 核心头文件
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/PlannerTerminationCondition.h>   // 终止条件
// 在现有的 OMPL 头文件包含之后，添加这一行
#include <ompl/geometric/PathGeometric.h>
#include "itkImage.h"
#include "common.h"

using SDFImageType = itk::Image<float, 3>;

namespace ob = ompl::base;
namespace og = ompl::geometric;

// =========================================================================
// 1. 内部私有：OMPL 状态有效性检查器（ITK 3D SDF 桥接）
// =========================================================================
class OMPLValidityChecker : public ob::StateValidityChecker {
public:
    OMPLValidityChecker(const ob::SpaceInformationPtr& si, 
                         SDFImageType::Pointer sdfImage, 
                         float safety_margin)
        : ob::StateValidityChecker(si), sdfImage_(sdfImage), safety_margin_(safety_margin) {}

    bool isValid(const ob::State* state) const override {
        const auto* se3state = state->as<ob::SE3StateSpace::StateType>();
        const auto* pos = se3state->as<ob::RealVectorStateSpace::StateType>(0);
        
        SDFImageType::PointType pt;
        pt[0] = pos->values[0]; 
        pt[1] = pos->values[1]; 
        pt[2] = pos->values[2];

        SDFImageType::IndexType idx;
        if (sdfImage_->TransformPhysicalPointToIndex(pt, idx)) {
            return sdfImage_->GetPixel(idx) >= safety_margin_;
        }
        return false;
    }

private:
    SDFImageType::Pointer sdfImage_;
    float safety_margin_;
};

// =========================================================================
// 2. 外部接口：OMPL BIT* 独立规划函数（修正版）
// =========================================================================
inline std::vector<Node> plan_with_ompl_bitstar(
    const Node& start_pt,
    const Node& goal_pt,
    SDFImageType::Pointer sdfImage,
    double solve_time_limit = 3.0,
    int samples_per_batch = 200,
    double rewire_factor = 1.2,
    float safety_margin = 10.0f)
{
    std::vector<Node> final_path;

    // A. 获取 SDF 物理边界
    SDFImageType::RegionType region = sdfImage->GetLargestPossibleRegion();
    SDFImageType::SizeType size = region.GetSize();
    SDFImageType::PointType origin = sdfImage->GetOrigin();
    SDFImageType::SpacingType spacing = sdfImage->GetSpacing();
    
    double min_x = origin[0]; double max_x = origin[0] + size[0] * spacing[0];
    double min_y = origin[1]; double max_y = origin[1] + size[1] * spacing[1];
    double min_z = origin[2]; double max_z = origin[2] + size[2] * spacing[2];

    // B. 构建 SE3 状态空间
    auto space(std::make_shared<ob::SE3StateSpace>());
    ob::RealVectorBounds bounds(3);
    bounds.setLow(0, min_x); bounds.setHigh(0, max_x);
    bounds.setLow(1, min_y); bounds.setHigh(1, max_y);
    bounds.setLow(2, min_z); bounds.setHigh(2, max_z);
    space->setBounds(bounds);

    // C. 空间信息 + 碰撞检测
    auto si(std::make_shared<ob::SpaceInformation>(space));
    si->setStateValidityChecker(std::make_shared<OMPLValidityChecker>(si, sdfImage, safety_margin));
    space->setLongestValidSegmentFraction(0.01);
    si->setup();

    // D. 起点与终点
    ob::ScopedState<ob::SE3StateSpace> start(space);
    start->setX(start_pt.x); start->setY(start_pt.y); start->setZ(start_pt.z);
    start->rotation().setIdentity();

    ob::ScopedState<ob::SE3StateSpace> goal(space);
    goal->setX(goal_pt.x); goal->setY(goal_pt.y); goal->setZ(goal_pt.z);
    goal->rotation().setIdentity();

    // E. 问题定义与优化目标（最短路径）
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
    pdef->setStartAndGoalStates(start, goal);
    pdef->setOptimizationObjective(std::make_shared<ob::PathLengthOptimizationObjective>(si));

    // F. 实例化 BIT* 并配置参数
    auto planner = std::make_shared<og::BITstar>(si);
    planner->setSamplesPerBatch(samples_per_batch);
    planner->setRewireFactor(rewire_factor);

    // 不再设置 setTargetObjectiveFraction（不存在），改用组合终止条件

    planner->setProblemDefinition(pdef);
    planner->setup();

    // G. 构建终止条件：时间限制 或 目标成本达成
    // 计算目标成本（直线距离的 1.02 倍）
    double lower_bound = std::sqrt(
        std::pow(start_pt.x - goal_pt.x, 2) +
        std::pow(start_pt.y - goal_pt.y, 2) +
        std::pow(start_pt.z - goal_pt.z, 2)
    );
    ob::Cost target_cost(lower_bound * 1.02);

    // 时间终止条件
    ob::PlannerTerminationCondition time_condition = 
        ob::timedPlannerTerminationCondition(solve_time_limit);

    // 成本终止条件（当 bestCost <= target_cost 时停止）
    // 注意：需要捕获 planner 和 target_cost 的引用（或值）
    ob::PlannerTerminationCondition cost_condition(
        [planner, target_cost]() -> bool {
            // 当规划器已有解且其成本低于目标成本时返回 true
            return (planner->bestCost().value() <= target_cost.value());
        }
    );

    // 合并条件（任一满足即停止）
    ob::PlannerTerminationCondition combined_condition = 
        ob::plannerOrTerminationCondition(time_condition, cost_condition);

    // 启动规划（使用组合终止条件）
    ob::PlannerStatus solved = planner->solve(combined_condition);

    // H. 提取并转换路径
    if (solved) {
        og::PathGeometric path = *std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
        path.interpolate();

        final_path.reserve(path.getStateCount());
        for (size_t i = 0; i < path.getStateCount(); ++i) {
            const auto* se3state = path.getState(i)->as<ob::SE3StateSpace::StateType>();
            const auto* pos = se3state->as<ob::RealVectorStateSpace::StateType>(0);
            final_path.emplace_back(pos->values[0], pos->values[1], pos->values[2]);
        }
    } else {
        std::cerr << " [OMPL 封装函数警告] BIT* 规划器在限制时间内未找到可行安全航线！" << std::endl;
    }

    return final_path;
}

// =========================================================================
// 3. 外部接口：OMPL AIT* 独立规划函数
// =========================================================================
inline std::vector<Node> plan_with_ompl_aitstar(
    const Node& start_pt,
    const Node& goal_pt,
    SDFImageType::Pointer sdfImage,
    double solve_time_limit = 3.0,
    std::size_t batch_size = 100,           // AIT* 每批处理的样本数量
    double rewire_factor = 1.0,            // 重连因子
    float safety_margin = 10.0f)
{
    std::vector<Node> final_path;

    // A. 获取 SDF 物理边界
    SDFImageType::RegionType region = sdfImage->GetLargestPossibleRegion();
    SDFImageType::SizeType size = region.GetSize();
    SDFImageType::PointType origin = sdfImage->GetOrigin();
    SDFImageType::SpacingType spacing = sdfImage->GetSpacing();

    double min_x = origin[0]; double max_x = origin[0] + size[0] * spacing[0];
    double min_y = origin[1]; double max_y = origin[1] + size[1] * spacing[1];
    double min_z = origin[2]; double max_z = origin[2] + size[2] * spacing[2];

    // B. 构建 SE3 状态空间
    auto space(std::make_shared<ob::SE3StateSpace>());
    ob::RealVectorBounds bounds(3);
    bounds.setLow(0, min_x); bounds.setHigh(0, max_x);
    bounds.setLow(1, min_y); bounds.setHigh(1, max_y);
    bounds.setLow(2, min_z); bounds.setHigh(2, max_z);
    space->setBounds(bounds);

    // C. 空间信息 + 碰撞检测
    auto si(std::make_shared<ob::SpaceInformation>(space));
    si->setStateValidityChecker(std::make_shared<OMPLValidityChecker>(si, sdfImage, safety_margin));
    space->setLongestValidSegmentFraction(0.01);
    si->setup();

    // D. 起点与终点
    ob::ScopedState<ob::SE3StateSpace> start(space);
    start->setX(start_pt.x); start->setY(start_pt.y); start->setZ(start_pt.z);
    start->rotation().setIdentity();

    ob::ScopedState<ob::SE3StateSpace> goal(space);
    goal->setX(goal_pt.x); goal->setY(goal_pt.y); goal->setZ(goal_pt.z);
    goal->rotation().setIdentity();

    // E. 问题定义与优化目标（最短路径）
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
    pdef->setStartAndGoalStates(start, goal);
    pdef->setOptimizationObjective(std::make_shared<ob::PathLengthOptimizationObjective>(si));

    // F. 实例化 AIT* 并配置参数
    auto planner = std::make_shared<og::AITstar>(si);   // 使用 AITstar 类[reference:8]

    // AIT* 配置参数[reference:9]
    planner->setBatchSize(batch_size);                  // 每批样本数[reference:10]
    planner->setRewireFactor(rewire_factor);            // 重连因子[reference:11]

    planner->setProblemDefinition(pdef);
    planner->setup();

    // G. 构建终止条件：时间限制 或 目标成本达成
    // 计算目标成本（直线距离的 1.02 倍）
    double lower_bound = std::sqrt(
        std::pow(start_pt.x - goal_pt.x, 2) +
        std::pow(start_pt.y - goal_pt.y, 2) +
        std::pow(start_pt.z - goal_pt.z, 2)
    );
    ob::Cost target_cost(lower_bound * 1.02);

    // 时间终止条件
    ob::PlannerTerminationCondition time_condition =
        ob::timedPlannerTerminationCondition(solve_time_limit);

    // 成本终止条件（当 bestCost <= target_cost 时停止）[reference:12]
    ob::PlannerTerminationCondition cost_condition(
        [planner, target_cost]() -> bool {
            return (planner->bestCost().value() <= target_cost.value());
        }
    );

    // 合并条件（任一满足即停止）
    ob::PlannerTerminationCondition combined_condition =
        ob::plannerOrTerminationCondition(time_condition, cost_condition);

    // 启动规划[reference:13]
    ob::PlannerStatus solved = planner->solve(combined_condition);

    // H. 提取并转换路径
    if (solved) {
        og::PathGeometric path = *std::static_pointer_cast<og::PathGeometric>(pdef->getSolutionPath());
        path.interpolate();

        final_path.reserve(path.getStateCount());
        for (std::size_t i = 0; i < path.getStateCount(); ++i) {
            const auto* se3state = path.getState(i)->as<ob::SE3StateSpace::StateType>();
            const auto* pos = se3state->as<ob::RealVectorStateSpace::StateType>(0);
            final_path.emplace_back(pos->values[0], pos->values[1], pos->values[2]);
        }
    } else {
        std::cerr << " [OMPL 封装函数警告] AIT* 规划器在限制时间内未找到可行安全航线！" << std::endl;
    }

    return final_path;
}