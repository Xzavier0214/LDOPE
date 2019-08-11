#ifndef LDOPE_UTIL_H
#define LDOPE_UTIL_H

#include <string.h>

constexpr size_t g_DOUBLE_2 = 2 * sizeof(double);
constexpr size_t g_DOUBLE_3 = 3 * sizeof(double);
constexpr size_t g_DOUBLE_6 = 6 * sizeof(double);
constexpr size_t g_DOUBLE_12 = 12 * sizeof(double);
constexpr size_t g_DOUBLE_24 = 24 * sizeof(double);

//////////////////////////球坐标系//////////////////////////

constexpr size_t SPHERICAL_STATE_SIZE = 6;
constexpr size_t SPHERICAL_CONTROL_SIZE = 2;
constexpr size_t SPHERICAL_EXT_STATE_SIZE = 4 * SPHERICAL_STATE_SIZE;
constexpr size_t SPHERICAL_BOUNDARY_SIZE = 13;
constexpr size_t SPHERICAL_INDIVIDUAL_SIZE = 11;

//  球坐标系个体转换
void SphericalIndividualConvertFcn(const double *pIndividual, double *pCostateP,
                                   double *pCostateE, double &tf);

//  球坐标系归一化
void SphericalStateNormFcn(const double *pState, double du, double tu, double *pNormedState);

//  球坐标系反归一化
void SphericalStateDenormFcn(const double *pState, double du, double tu, double *pDenormedState);

//////////////////////////笛卡尔坐标系//////////////////////////

constexpr size_t CARTESIAN_STATE_SIZE = 6;
constexpr size_t CARTESIAN_CONTROL_SIZE = 2;
constexpr size_t CARTESIAN_EXT_STATE_SIZE = 4 * CARTESIAN_STATE_SIZE;
constexpr size_t CARTESIAN_BOUNDARY_SIZE = 13;
constexpr size_t CARTESIAN_INDIVIDUAL_SIZE = 13;

//  笛卡尔坐标系个体转换
void CartesianIndividualConvertFcn(const double *pIndividual, double *pCostateP,
                                   double *pCostateE, double &tf);

//  笛卡尔坐标系归一化
void CartesianStateNormFcn(const double *pState, double du, double tu, double *pNormedState);

//  笛卡尔坐标反归一化
void CartesianStateDenormFcn(const double *pState, double du, double tu, double *pDenormedState);

//  导出
extern "C"
{
    //  球坐标系常量
    void get_spherical_state_size(size_t &stateSize);
    void get_spherical_control_size(size_t &controlSize);
    void get_spherical_ext_state_size(size_t &extStateSize);
    void get_spherical_boundary_size(size_t &boundarySize);
    void get_spherical_individual_size(size_t &individualSize);

    //  球坐标系个体转换
    void spherical_individual_convert_fcn(const double *pIndividual, double *pCostateP,
                                          double *pCostateE, double &tf);

    //  球坐标系归一化
    void spherical_state_norm_fcn(const double *pState, double du, double tu,
                                  double *pNormedState);

    //  球坐标系反归一化
    void spherical_state_denorm_fcn(const double *pState, double du, double tu,
                                    double *pDenormedState);

    //  笛卡尔坐标常量
    void get_cartesian_state_size(size_t &stateSize);
    void get_cartesian_control_size(size_t &controlSize);
    void get_cartesian_ext_state_size(size_t &extStateSize);
    void get_cartesian_boundary_size(size_t &boundarySize);
    void get_cartesian_individual_size(size_t &individualSize);

    //  笛卡尔坐标个体转换
    void cartesian_individual_convert_fcn(const double *pIndividual, double *pCostateP,
                                          double *pCostateE, double &tf);

    //  笛卡尔坐标归一化
    void cartesian_state_norm_fcn(const double *pState, double du, double tu,
                                  double *pNormedState);

    //  笛卡尔坐标反归一化
    void cartesian_state_denorm_fcn(const double *pState, double du, double tu,
                                    double *pDenormedState);
}

#endif //LDOPE_UTIL_H
