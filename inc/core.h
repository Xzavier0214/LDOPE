#ifndef LDOPE_CORE_H
#define LDOPE_CORE_H

#include <string.h>

struct Param {
    double *pInitialStateP; // 追踪者初始状态（非归一化）
    double *pInitialStateE; // 逃逸者初始状态（非归一化）
    double tmP;             // 追踪者推重比（非归一化）
    double tmE;             // 逃逸者推重比（非归一化）
    double *lb;             // 个体下限（归一化）
    double *ub;             // 个体上限（归一化）
    double *pK;             // 权重系数
    double du;              // 距离归一化系数（非归一化）
    double tu;              // 时间归一化系数（非归一化）

    unsigned int dim;       // 个体维数（Spherical：11，Cartesian：13）
    bool printProcess;      // 打印计算过程
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
double SphericalHamiltonFcn(const double *pState, const double *pCostate, const double *pControl,
                            double tm, double mu);

//  球坐标系扩展状态微分方程
void SphericalExtStateFcn(const double *pExtState, double tmP, double tmE, double mu, double *pDotExtState);

//  球坐标系边界条件
void SphericalBoundaryFcn(const double *pExtState, double tmP, double tmE, double mu, double *pBoundary);

//  球坐标系适应度函数
double SphericalFitnessFcn(const double *pIndividual, const double *pK, const double *pStateP0, const double *pStateE0,
                           double tmP, double tmE, double du, double tu);

//  求解球坐标系
double SphericalSolveFcn(Param *pParam, double *pOptIndividual, bool printProcess = false);

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
double CartesianHamiltonFcn(const double *pState, const double *pCostate, const double *pControl,
                            double tm, double mu);

//  笛卡尔坐标系扩展状态微分方程
void CartesianExtStateFcn(const double *pExtState, double tmP, double tmE, double mu, double *pDotExtState);

//  笛卡尔坐标系边界条件
void CartesianBoundaryFcn(const double *pExtState, double tmP, double tmE, double mu, double *pBoundary);

//  笛卡尔坐标系适应度函数
double CartesianFitnessFcn(const double *pIndividual, const double *pK, const double *pStateP0, const double *pStateE0,
                           double tmP, double tmE, double du, double tu);

//  求解笛卡尔坐标系
double CartesianSolveFcn(Param *pParam, double *pOptIndividual, bool printProcess = false);

#endif //LDOPE_CORE_H
