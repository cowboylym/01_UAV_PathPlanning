#pragma once
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/geometric/planners/informedtrees/BITstar.h>
#include <ompl/geometric/planners/informedtrees/AITstar.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/PlannerTerminationCondition.h>
#include <ompl/geometric/PathGeometric.h>
#include "dem_collision.hpp"
#include "common.h"

namespace ob = ompl::base;
namespace og = ompl::geometric;

class OMPLValidityChecker_DEM : public ob::StateValidityChecker {
public:
    OMPLValidityChecker_DEM(const ob::SpaceInformationPtr& si,
                            const DEMData& dem,
                            double safety_margin)
        : ob::StateValidityChecker(si), dem_(dem), safety_margin_(safety_margin) {}

    bool isValid(const ob::State* state) const override {
        const auto* se3state = state->as<ob::SE3StateSpace::StateType>();
        const auto* pos = se3state->as<ob::RealVectorStateSpace::StateType>(0);
        return is_state_valid_dem(pos->values[0], pos->values[1], pos->values[2],
                                  dem_, safety_margin_);
    }
private:
    const DEMData& dem_;
    double safety_margin_;
};

// BIT* DEM 版本
inline std::vector<Node> plan_with_ompl_bitstar_dem(
    const Node& start_pt, const Node& goal_pt,
    const DEMData& dem,
    double solve_time_limit = 3.0,
    int samples_per_batch = 200,
    double rewire_factor = 1.2,
    double safety_margin = 10.0)
{
    std::vector<Node> final_path;
    // 定义状态空间边界 (从 DEM 获取)
    double min_x = dem.geoTransform[0];
    double max_x = dem.geoTransform[0] + dem.width * dem.geoTransform[1];
    double min_y = dem.geoTransform[3] + dem.height * dem.geoTransform[5];
    double max_y = dem.geoTransform[3];
    // Z 范围需用户设定或从 DEM 高程计算，这里简单设为 0~500
    double min_z = 0.0, max_z = 500.0;

    auto space(std::make_shared<ob::SE3StateSpace>());
    ob::RealVectorBounds bounds(3);
    bounds.setLow(0, min_x); bounds.setHigh(0, max_x);
    bounds.setLow(1, min_y); bounds.setHigh(1, max_y);
    bounds.setLow(2, min_z); bounds.setHigh(2, max_z);
    space->setBounds(bounds);

    auto si(std::make_shared<ob::SpaceInformation>(space));
    si->setStateValidityChecker(std::make_shared<OMPLValidityChecker_DEM>(si, dem, safety_margin));
    space->setLongestValidSegmentFraction(0.01);
    si->setup();

    ob::ScopedState<ob::SE3StateSpace> start(space);
    start->setX(start_pt.x); start->setY(start_pt.y); start->setZ(start_pt.z);
    start->rotation().setIdentity();

    ob::ScopedState<ob::SE3StateSpace> goal(space);
    goal->setX(goal_pt.x); goal->setY(goal_pt.y); goal->setZ(goal_pt.z);
    goal->rotation().setIdentity();

    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
    pdef->setStartAndGoalStates(start, goal);
    pdef->setOptimizationObjective(std::make_shared<ob::PathLengthOptimizationObjective>(si));

    auto planner = std::make_shared<og::BITstar>(si);
    planner->setSamplesPerBatch(samples_per_batch);
    planner->setRewireFactor(rewire_factor);
    planner->setProblemDefinition(pdef);
    planner->setup();

    // 终止条件：时间限制
    ob::PlannerTerminationCondition time_condition = ob::timedPlannerTerminationCondition(solve_time_limit);
    ob::PlannerStatus solved = planner->solve(time_condition);

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
        std::cerr << "[BIT* DEM] 规划失败\n";
    }
    return final_path;
}

// AIT* DEM 版本 (与 BIT* 类似，仅 planner 类型不同)
inline std::vector<Node> plan_with_ompl_aitstar_dem(
    const Node& start_pt, const Node& goal_pt,
    const DEMData& dem,
    double solve_time_limit = 3.0,
    std::size_t batch_size = 100,
    double rewire_factor = 1.0,
    double safety_margin = 10.0)
{
    std::vector<Node> final_path;
    // 边界同 BIT*
    double min_x = dem.geoTransform[0];
    double max_x = dem.geoTransform[0] + dem.width * dem.geoTransform[1];
    double min_y = dem.geoTransform[3] + dem.height * dem.geoTransform[5];
    double max_y = dem.geoTransform[3];
    double min_z = 0.0, max_z = 500.0;

    auto space(std::make_shared<ob::SE3StateSpace>());
    ob::RealVectorBounds bounds(3);
    bounds.setLow(0, min_x); bounds.setHigh(0, max_x);
    bounds.setLow(1, min_y); bounds.setHigh(1, max_y);
    bounds.setLow(2, min_z); bounds.setHigh(2, max_z);
    space->setBounds(bounds);

    auto si(std::make_shared<ob::SpaceInformation>(space));
    si->setStateValidityChecker(std::make_shared<OMPLValidityChecker_DEM>(si, dem, safety_margin));
    space->setLongestValidSegmentFraction(0.01);
    si->setup();

    ob::ScopedState<ob::SE3StateSpace> start(space);
    start->setX(start_pt.x); start->setY(start_pt.y); start->setZ(start_pt.z);
    start->rotation().setIdentity();
    ob::ScopedState<ob::SE3StateSpace> goal(space);
    goal->setX(goal_pt.x); goal->setY(goal_pt.y); goal->setZ(goal_pt.z);
    goal->rotation().setIdentity();

    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
    pdef->setStartAndGoalStates(start, goal);
    pdef->setOptimizationObjective(std::make_shared<ob::PathLengthOptimizationObjective>(si));

    auto planner = std::make_shared<og::AITstar>(si);
    planner->setBatchSize(batch_size);
    planner->setRewireFactor(rewire_factor);
    planner->setProblemDefinition(pdef);
    planner->setup();

    ob::PlannerTerminationCondition time_condition = ob::timedPlannerTerminationCondition(solve_time_limit);
    ob::PlannerStatus solved = planner->solve(time_condition);

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
        std::cerr << "[AIT* DEM] 规划失败\n";
    }
    return final_path;
}