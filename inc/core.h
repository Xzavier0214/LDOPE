#ifndef LDOPE_CORE_H
#define LDOPE_CORE_H

#include <string.h>

struct Param
{
    double *pInitialStateP; // 追踪者初始状态（非归一化）
    double *pInitialStateE; // 逃逸者初始状态（非归一化）
    double tmP;             // 追踪者推重比（非归一化）
    double tmE;             // 逃逸者推重比（非归一化）
    double *lb;             // 个体下限（归一化）
    double *ub;             // 个体上限（归一化）
    double *pK;             // 权重系数
    double du;              // 距离归一化系数（非归一化）
    double tu;              // 时间归一化系数（非归一化）
    bool printProcess;      // 打印计算过程
    bool normProcess;       // 归一化计算过程，仅对笛卡尔模型有效
};

//////////////////////////球坐标系//////////////////////////

//  球坐标系状态微分方程
void SphericalStateFcn(const double *pState, const double *pControl,
                       double tm, double mu, double *pDotState);

//  球坐标系协态微分方程
void SphericalCostateFcn(const double *pState, const double *pCostate, const double *pControl,
                         double tm, double mu, double *pDotCostate);

//  球坐标系控制变量函数
void SphericalControlFcn(const double *pState, const double *pCostate, int flag, double *pControl);

//  球坐标系哈密顿函数
void SphericalHamiltonFcn(const double *pState, const double *pCostate, const double *pControl,
                          double tm, double mu, double &hamilton);

//  球坐标系扩展状态微分方程
void SphericalExtStateFcn(const double *pExtState, double tmP, double tmE, double mu, double *pDotExtState);

//  球坐标系边界条件
void SphericalBoundaryFcn(const double *pExtState, double tmP, double tmE, double mu, double *pBoundary);

//  球坐标系适应度函数
void SphericalFitnessFcn(const double *pIndividual, const double *pK, const double *pStateP0, const double *pStateE0,
                         double tmP, double tmE, double du, double tu, double &fitness);

//  求解球坐标系
void SphericalSolveFcn(Param *pParam, double *pOptIndividual, double &optValue);

//////////////////////////笛卡尔坐标系//////////////////////////

//  笛卡尔坐标系状态微分方程
void CartesianStateFcn(const double *pState, const double *pControl,
                       double tm, double mu, double *pDotState);

//  笛卡尔坐标系协态微分方程
void CartesianCostateFcn(const double *pState, const double *pCostate, const double *pControl,
                         double tm, double mu, double *pDotCostate);

//  笛卡尔坐标系控制变量函数
void CartesianControlFcn(const double *pState, const double *pCostate, int flag, double *pControl);

//  笛卡尔坐标系哈密顿函数
void CartesianHamiltonFcn(const double *pState, const double *pCostate, const double *pControl,
                          double tm, double mu, double &hamilton);

//  笛卡尔坐标系扩展状态微分方程
void CartesianExtStateFcn(const double *pExtState, double tmP, double tmE, double mu, double *pDotExtState);

//  笛卡尔坐标系边界条件
void CartesianBoundaryFcn(const double *pExtState, double tmP, double tmE, double mu, double *pBoundary);

//  笛卡尔坐标系适应度函数
void CartesianFitnessFcn(const double *pIndividual, const double *pK, const double *pStateP0, const double *pStateE0,
                         double tmP, double tmE, double du, double tu, double &fitness, bool normProcess=true);

//  求解笛卡尔坐标系
void CartesianSolveFcn(Param *pParam, double *pOptIndividual, double &optValue);

//  导出
extern "C"
{
    //  球坐标系状态微分方程
    void spherical_state_fcn(const double *pState, const double *pControl,
                             double tm, double mu, double *pDotState);

    //  球坐标系协态微分方程
    void spherical_costate_fcn(const double *pState, const double *pCostate, const double *pControl,
                               double tm, double mu, double *pDotCostate);

    //  球坐标系控制变量函数
    void spherical_control_fcn(const double *pState, const double *pCostate, int flag, double *pControl);

    //  球坐标系哈密顿函数
    void spherical_hamilton_fcn(const double *pState, const double *pCostate, const double *pControl,
                                double tm, double mu, double &hamilton);

    //  球坐标系扩展状态微分方程
    void spherical_ext_state_fcn(const double *pExtState, double tmP, double tmE, double mu, double *pDotExtState);

    //  球坐标系边界条件
    void spherical_boundary_fcn(const double *pExtState, double tmP, double tmE, double mu, double *pBoundary);

    //  球坐标系适应度函数
    void spherical_fitness_fcn(const double *pIndividual, const double *pK, const double *pStateP0, const double *pStateE0,
                               double tmP, double tmE, double du, double tu, double &fitness);

    //  求解球坐标系
    void spherical_solve_fcn(Param *pParam, double *pOptIndividual, double &optValue);

    //  笛卡尔坐标系状态微分方程
    void cartesian_state_fcn(const double *pState, const double *pControl,
                             double tm, double mu, double *pDotState);

    //  笛卡尔坐标系协态微分方程
    void cartesian_costate_fcn(const double *pState, const double *pCostate, const double *pControl,
                               double tm, double mu, double *pDotCostate);

    //  笛卡尔坐标系控制变量函数
    void cartesian_control_fcn(const double *pState, const double *pCostate, int flag, double *pControl);

    //  笛卡尔坐标系哈密顿函数
    void cartesian_hamilton_fcn(const double *pState, const double *pCostate, const double *pControl,
                                double tm, double mu, double &hamilton);

    //  笛卡尔坐标系扩展状态微分方程
    void cartesian_ext_state_fcn(const double *pExtState, double tmP, double tmE, double mu, double *pDotExtState);

    //  笛卡尔坐标系边界条件
    void cartesian_boundary_fcn(const double *pExtState, double tmP, double tmE, double mu, double *pBoundary);

    //  笛卡尔坐标系适应度函数
    void cartesian_fitness_fcn(const double *pIndividual, const double *pK, const double *pStateP0, const double *pStateE0,
                               double tmP, double tmE, double du, double tu, double &fitness, bool norm_process=true);

    //  求解笛卡尔坐标系
    void cartesian_solve_fcn(Param *pParam, double *pOptIndividual, double &optValue);
}

#endif //LDOPE_CORE_H
