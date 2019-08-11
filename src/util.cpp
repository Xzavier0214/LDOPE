#include "util.h"

//////////////////////////球坐标系//////////////////////////

void SphericalIndividualConvertFcn(const double *pIndividual, double *pCostateP,
                                   double *pCostateE, double &tf)
{
    memcpy(pCostateP, pIndividual, g_DOUBLE_3);
    pCostateP[3] = 0;
    memcpy(pCostateP + 4, pIndividual + 3, g_DOUBLE_2);
    memcpy(pCostateE, pIndividual + 5, g_DOUBLE_3);
    pCostateE[3] = 0;
    memcpy(pCostateE + 4, pIndividual + 8, g_DOUBLE_2);
    tf = pIndividual[10];
}

void SphericalStateNormFcn(const double *pState, double du, double tu, double *pNormedState)
{
    memcpy(pNormedState, pState, g_DOUBLE_6);
    double vu = du / tu;
    pNormedState[0] /= du;
    pNormedState[1] /= vu;
}

void SphericalStateDenormFcn(const double *pState, double du, double tu, double *pDenormedState)
{
    memcpy(pDenormedState, pState, g_DOUBLE_6);
    double vu = du / tu;
    pDenormedState[0] *= du;
    pDenormedState[1] *= vu;
}

//////////////////////////笛卡尔坐标系//////////////////////////

void CartesianIndividualConvertFcn(const double *pIndividual, double *pCostateP,
                                   double *pCostateE, double &tf)
{
    memcpy(pCostateP, pIndividual, g_DOUBLE_6);
    memcpy(pCostateE, pIndividual + 6, g_DOUBLE_6);
    tf = pIndividual[12];
}

void CartesianStateNormFcn(const double *pState, double du, double tu, double *pNormedState)
{
    for (int i = 0; i < 3; i++)
    {
        double vu = du / tu;
        pNormedState[i] = pState[i] / du;
        pNormedState[3 + i] = pState[3 + i] / vu;
    }
}

void CartesianStateDenormFcn(const double *pState, double du, double tu, double *pDenormedState)
{
    for (int i = 0; i < 3; i++)
    {
        double vu = du / tu;
        pDenormedState[i] = pState[i] * du;
        pDenormedState[3 + i] = pState[3 + i] * vu;
    }
}

//////////////////////////导出//////////////////////////

extern "C"
{
    //  球坐标系常量
    size_t get_spherical_state_size() { return SPHERICAL_STATE_SIZE; }
    size_t get_spherical_control_size() { return SPHERICAL_CONTROL_SIZE; }
    size_t get_spherical_ext_state_size() { return SPHERICAL_EXT_STATE_SIZE; }
    size_t get_spherical_boundary_size() { return SPHERICAL_BOUNDARY_SIZE; }
    size_t get_spherical_individual_size() { return SPHERICAL_INDIVIDUAL_SIZE; }

    //  球坐标系个体转换
    void spherical_individual_convert_fcn(const double *pIndividual, double *pCostateP,
                                          double *pCostateE, double &tf)
    {
        SphericalIndividualConvertFcn(pIndividual, pCostateP, pCostateE, tf);
    }

    //  球坐标系归一化
    void spherical_state_norm_fcn(const double *pState, double du, double tu,
                                  double *pNormedState)
    {
        SphericalStateNormFcn(pState, du, tu, pNormedState);
    }

    //  球坐标系反归一化
    void spherical_state_denorm_fcn(const double *pState, double du, double tu,
                                    double *pDenormedState)
    {
        SphericalStateDenormFcn(pState, du, tu, pDenormedState);
    }

    //  笛卡尔坐标系常量
    size_t get_cartesian_state_size() { return CARTESIAN_STATE_SIZE; }
    size_t get_cartesian_control_size() { return CARTESIAN_CONTROL_SIZE; }
    size_t get_cartesian_ext_state_size() { return CARTESIAN_EXT_STATE_SIZE; }
    size_t get_cartesian_boundary_size() { return CARTESIAN_BOUNDARY_SIZE; }
    size_t get_cartesian_individual_size() { return CARTESIAN_INDIVIDUAL_SIZE; }

    //  笛卡尔坐标系个体转换
    void cartesian_individual_convert_fcn(const double *pIndividual, double *pCostateP,
                                          double *pCostateE, double &tf)
    {
        CartesianIndividualConvertFcn(pIndividual, pCostateP, pCostateE, tf);
    }

    //  笛卡尔坐标系归一化
    void cartesian_state_norm_fcn(const double *pState, double du, double tu,
                                  double *pNormedState)
    {
        CartesianStateNormFcn(pState, du, tu, pNormedState);
    }

    //  笛卡尔坐标系反归一化
    void cartesian_state_denorm_fcn(const double *pState, double du, double tu,
                                    double *pDenormedState)
    {
        CartesianStateDenormFcn(pState, du, tu, pDenormedState);
    }
}
