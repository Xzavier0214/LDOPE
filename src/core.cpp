#include <iostream>
#include <cstdlib>
#include <math.h>
#include <string.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>
#include <nlopt.hpp>
#include "util.h"
#include "core.h"
using namespace std;

//////////////////////////球坐标系//////////////////////////

void SphericalStateFcn(const double *pState, const double *pControl,
                       double tm, double mu, double *pDotState) {
    double r   = pState[0], v    = pState[1], gamma = pState[2];
    double phi = pState[4], zeta = pState[5];

    double sin_g = sin(gamma), cos_g = cos(gamma);
    double sin_p = sin(phi),   cos_p = cos(phi);
    double sin_z = sin(zeta),  cos_z = cos(zeta);

    pDotState[0] = v*sin_g;
    pDotState[1] = -mu*sin_g/(r*r);
    pDotState[2] = v*cos_g/r - mu*cos_g/(r*r*v);
    pDotState[3] = v*cos_g*cos_z/(r*cos_p);
    pDotState[4] = v*cos_g*sin_z/r;
    pDotState[5] = -v*cos_g*sin_p*cos_z/(r*cos_p);

    if (pControl != nullptr) {
        double alpha = pControl[0], beta = pControl[1];

        double sin_a = sin(alpha), cos_a = cos(alpha);
        double sin_b = sin(beta),  cos_b = cos(beta);

        pDotState[1] += tm*cos_a*cos_b;
        pDotState[2] += tm*sin_a*cos_b/v;
        pDotState[5] += tm*sin_b/(v*cos_g);
    }
}

void SphericalCostateFcn(const double *pState, const double *pCostate, const double *pControl,
                         double tm, double mu, double *pDotCostate) {
    double r = pState[0], v = pState[1];
    double gamma = pState[2], phi = pState[4], zeta = pState[5];
    double l1 = pCostate[0], l2 = pCostate[1], l3 = pCostate[2];
    double l4 = pCostate[3], l5 = pCostate[4], l6 = pCostate[5];
    double alpha = pControl[0], beta = pControl[1];

    double sin_g = sin(gamma), cos_g = cos(gamma);
    double sin_p = sin(phi),   cos_p = cos(phi);
    double sin_z = sin(zeta),  cos_z = cos(zeta);
    double sin_a = sin(alpha);
    double sin_b = sin(beta),  cos_b = cos(beta);

    double r2 = r*r, r3 = r2*r, v2 = v*v;

    pDotCostate[0] = (v*l3*cos_g + l4*v*cos_g*cos_z/cos_p + v*l5*cos_g*sin_z - v*l6*cos_g*sin_p*cos_z/cos_p)/r2 +
                     -(2*mu*l2*sin_g + 2*mu*l3*cos_g/v)/r3;
    pDotCostate[1] = -l1*sin_g - l3*(cos_g*(1/r + mu/(r2*v2)) - tm/v2*sin_a*cos_b) +
                     -l4*cos_g*cos_z/(r*cos_p) - l5*cos_g*sin_z/r +l6*(tm*sin_b/(v2*cos_g) + cos_g*sin_p*cos_z/(r*cos_p));
    pDotCostate[2] = -v*l1*cos_g + mu*l2*cos_g/r2 + l3*sin_g*(v/r - mu/(r2*v)) + v*l4*sin_g*cos_z/(r*cos_p) +
                     v*l5*sin_g*sin_z/r - l6*(tm*sin_b*sin_g/(v*cos_g*cos_g) + v*sin_g*sin_p*cos_z/(r*cos_p));
    pDotCostate[3] = 0;
    pDotCostate[4] = v*cos_g*cos_z/(r*cos_p*cos_p)*(-l4*sin_p + l6);
    pDotCostate[5] = v*cos_g/(r*cos_p)*(l4*sin_z - l5*cos_p*cos_z - l6*sin_p*sin_z);
}

void SphericalControlFcn(const double *pState, const double *pCostate, int flag, double *pControl) {
    double gAlpha[2];
    gAlpha[0] = atan2(pCostate[2], pState[1]*pCostate[1]);
    gAlpha[1] = gAlpha[0] > 0 ? gAlpha[0] - M_PI : gAlpha[0] + M_PI;

    auto beta_fcn = [=](double alpha, double *pBeta) {
        double beta1 = atan2(pCostate[5], cos(pState[2])*(pCostate[2]*sin(alpha) + pCostate[1]*pState[1]*cos(alpha)));
        double beta2 = beta1 + M_PI;
        if (beta2 > M_PI) beta2 = beta1 - M_PI;
        pBeta[0] = beta1;
        pBeta[1] = beta2;
    };

    auto check_fcn = [=](double alpha, double beta) {
        double sin_a = sin(alpha), cos_a = cos(alpha);
        double sin_b = sin(beta),  cos_b = cos(beta);

        double h11 = -cos_b/pState[1]*(pState[1]*pCostate[1]*cos_a + pCostate[2]*sin_a);
        double h22 = -(1/pState[1]*(pCostate[2]*sin_a*cos_b + pCostate[5]*sin_b/cos(pState[2])) + pCostate[1]*cos_a*cos_b);
        double h12 = sin_b/pState[1]*(pState[1]*pCostate[1]*sin_a - pCostate[2]*cos_a);
        double h21 = h12;

        return flag*(h11 + h22) < 0 && h11*h22 - h12*h21 > 0;
    };

    for (auto alpha : gAlpha) {
        pControl[0] = alpha;
        double gBeta[2];
        beta_fcn(alpha, gBeta);
        for (auto beta : gBeta) {
            if (check_fcn(alpha, beta)) {
                pControl[1] = beta;
                return;
            }
        }
    }
}

double SphericalHamiltonFcn(const double *pState, const double *pCostate, const double *pControl,
                            double tm, double mu) {
    double gDotState[6];
    SphericalStateFcn(pState, pControl, tm, mu, gDotState);

    double result = .0;
    for (int i = 0; i < 6; i++) result += pCostate[i]*gDotState[i];
    return result;
}

void SphericalExtStateFcn(const double *pExtState, double tmP, double tmE, double mu, double *pDotExtState) {
    const double *pStateP   = pExtState;
    const double *pStateE   = pExtState + 6;
    const double *pCostateP = pExtState + 12;
    const double *pCostateE = pExtState + 18;

    double controlP[2], controlE[2];
    SphericalControlFcn(pStateP, pCostateP, -1, controlP);
    SphericalControlFcn(pStateE, pCostateE, 1,  controlE);

    SphericalStateFcn(pStateP, controlP, tmP, mu, pDotExtState);
    SphericalStateFcn(pStateE, controlE, tmE, mu, pDotExtState + 6);
    SphericalCostateFcn(pStateP, pCostateP, controlP, tmP, mu, pDotExtState + 12);
    SphericalCostateFcn(pStateE, pCostateE, controlE, tmE, mu, pDotExtState + 18);
}

void SphericalBoundaryFcn(const double *pExtState, double tmP, double tmE, double mu, double *pBoundary) {
    const double *pStateP   = pExtState;
    const double *pStateE   = pExtState + 6;
    const double *pCostateP = pExtState + 12;
    const double *pCostateE = pExtState + 18;

    pBoundary[0]  = pStateP[0]   - pStateE[0];
    pBoundary[1]  = pStateP[3]   - pStateE[3];
    pBoundary[2]  = pStateP[4]   - pStateE[4];
    pBoundary[3]  = pCostateP[0] + pCostateE[0];
    pBoundary[4]  = pCostateP[3] + pCostateE[3];
    pBoundary[5]  = pCostateP[4] + pCostateE[4];
    pBoundary[6]  = pCostateP[1];
    pBoundary[7]  = pCostateP[2];
    pBoundary[8]  = pCostateP[5];
    pBoundary[9]  = pCostateE[1];
    pBoundary[10] = pCostateE[2];
    pBoundary[11] = pCostateE[5];

    double gControlP[2], gControlE[2];
    SphericalControlFcn(pStateP, pCostateP, -1, gControlP);
    SphericalControlFcn(pStateE, pCostateE, 1,  gControlE);
    pBoundary[12] = 1 + SphericalHamiltonFcn(pStateP, pCostateP, gControlP, tmP, mu) +
                    SphericalHamiltonFcn(pStateE, pCostateE, gControlE, tmE, mu);
}

double SphericalFitnessFcn(const double *pIndividual, const double *pK, const double *pStateP0, const double *pStateE0,
                           double tmP, double tmE, double du, double tu) {
    static auto norm_fcn = [](const double *pExtState, double du, double tu, double *pNormExtState) {
        memcpy(pNormExtState, pExtState, g_DOUBLE_24);
        double vu = du/tu;
        pNormExtState[0] /= du;
        pNormExtState[6] /= du;
        pNormExtState[1] /= vu;
        pNormExtState[7] /= vu;
    };

    static auto denorm_fcn = [](const double *pExtState, double du, double tu, double *pDenormExtState) {
        memcpy(pDenormExtState, pExtState, g_DOUBLE_24);
        double vu = du/tu;
        pDenormExtState[0] *= du;
        pDenormExtState[6] *= du;
        pDenormExtState[1] *= vu;
        pDenormExtState[7] *= vu;
    };

    double gu = du/tu/tu;
    double mu = du*du*du/tu/tu;
    double param[2] = { tmP/gu, tmE/gu };

    gsl_odeiv2_system normOdeSystem = {
            [](double t, const double y[], double f[], void *params) -> int {
                (void)(t);
                double normTmP = *((double*)params);
                double normTmE = *((double*)params + 1);

                SphericalExtStateFcn(y, normTmP, normTmE, 1, f);

                return GSL_SUCCESS;
            }, nullptr, 24, param};
    gsl_odeiv2_driver *normOdeDriver =
            gsl_odeiv2_driver_alloc_y_new(&normOdeSystem, gsl_odeiv2_step_rkf45, 1e-3, 1e-6, 1e-6);

    double t = .0, tf, gExtState[24], gNormExtState[24];

    memcpy(gExtState,      pStateP0,        g_DOUBLE_6);
    memcpy(gExtState + 6,  pStateE0,        g_DOUBLE_6);
    SphericalIndividualConvertFcn(pIndividual, gExtState + 12, gExtState + 18, tf);
    norm_fcn(gExtState, du, tu, gNormExtState);

    int status = gsl_odeiv2_driver_apply(normOdeDriver, &t, tf, gNormExtState);
    if (status != GSL_SUCCESS) {
        printf ("error, return value=%d\n", status);
        exit(1);
    }
    else {
        denorm_fcn(gNormExtState, du, tu, gExtState);

        double gBoundary[13];
        SphericalBoundaryFcn(gExtState, tmP, tmE, mu, gBoundary);

        double target = .0;
        for (int i = 0; i < 13; i++) target += pK[i]*abs(gBoundary[i]);
        return target;
    }
}

double SphericalSolveFcn(Param *pParam, double *pOptIndividual, bool printProcess) {
    const unsigned int dim = 11;
    pParam->dim = dim;

    pParam->printProcess = printProcess;

    auto *pOptGlobal = new nlopt::opt(nlopt::algorithm::GN_ISRES, dim);
    auto *pOptLocalN = new nlopt::opt(nlopt::algorithm::LN_NELDERMEAD, dim);
//    auto *pOptLocalD = new nlopt::opt(nlopt::algorithm::LD_MMA, dim);

    auto optXGlobal = std::vector<double>(dim, 0);
    double optFGlobal = .0;

    auto optXLocalN = std::vector<double>(dim, 0);
    double optFLocalN = .0;

//    auto optXLocalD = std::vector<double>(dim, 0);
//    double optFLocalD = .0;

    for (int i = 0; i < dim; i++) {
        double temp = (pParam->lb[i] + pParam->ub[i])/2.0;
        optXGlobal[i] = temp;
        optXLocalN[i] = temp;
        // optXLocalD[i] = temp;
    }

    auto lb = std::vector<double>(pParam->lb, pParam->lb + dim);
    auto ub = std::vector<double>(pParam->ub, pParam->ub + dim);

    auto objective = [](const std::vector<double> &x, std::vector<double> &grad, void* f_data) {
        auto p = (Param*)f_data;
        auto result = SphericalFitnessFcn(x.data(), p->pK, p->pInitialStateP, p->pInitialStateE,
                                          p->tmP, p->tmE, p->du, p->tu);
        const unsigned int dim = p->dim;

//        if (!grad.empty()) {
//            double delta = 1e-8;
//            double tempX[dim];
//            double f1, f2;
//            memcpy(tempX, x.data(), dim*sizeof(double));
//            for (int i = 0; i < dim; i++) {
//                tempX[i] += delta/2;
//                f1 = SphericalFitnessFcn(tempX, p->pK, p->pInitialStateP, p->pInitialStateE,
//                                         p->tmP, p->tmE, p->du, p->tu);
//                tempX[i] -= delta;
//                f2 = SphericalFitnessFcn(tempX, p->pK, p->pInitialStateP, p->pInitialStateE,
//                                         p->tmP, p->tmE, p->du, p->tu);
//                tempX[i] += delta/2;
//                grad[i] = (f1 - f2)/delta;
//            }
//        }

        if (p->printProcess)
            cout << x.data()[dim - 1] << '\t' << result << endl;

        return result;
    };

    pOptGlobal->set_min_objective(objective, pParam);
    pOptGlobal->set_lower_bounds(lb);
    pOptGlobal->set_upper_bounds(ub);
    pOptGlobal->set_ftol_rel(1e-6);
    pOptGlobal->set_ftol_abs(1e-6);
    pOptGlobal->set_xtol_rel(1e-6);
    pOptGlobal->set_xtol_abs(1e-6);

    pOptLocalN->set_min_objective(objective, pParam);
    pOptLocalN->set_lower_bounds(lb);
    pOptLocalN->set_upper_bounds(ub);
    pOptLocalN->set_ftol_rel(1e-6);
    pOptLocalN->set_ftol_abs(1e-6);
    pOptLocalN->set_xtol_rel(1e-6);
    pOptLocalN->set_xtol_abs(1e-6);

//    pOptLocalD->set_min_objective(objective, pParam);
//    pOptLocalD->set_lower_bounds(lb);
//    pOptLocalD->set_upper_bounds(ub);
//    pOptLocalD->set_ftol_rel(1e-9);
//    pOptLocalD->set_ftol_abs(1e-9);
//    pOptLocalD->set_xtol_rel(1e-9);
//    pOptLocalD->set_xtol_abs(1e-9);

    if (printProcess)
        cout << "Global Optimization:" << endl;

    pOptGlobal->optimize(optXGlobal, optFGlobal);

    optXLocalN = optXGlobal;
    optFLocalN = optFGlobal;

    if (printProcess)
        cout << "Local Optimization:" << endl;

    pOptLocalN->optimize(optXLocalN, optFLocalN);

//    optXLocalD = optXLocalN;
//    optFLocalD = optFLocalN;

//    pOptLocalD->optimize(optXLocalD, optFLocalD);

//    memcpy(pOptIndividual, optXLocalD.data(), dim*sizeof(double));
//    return optFLocalD;

    memcpy(pOptIndividual, optXLocalN.data(), dim*sizeof(double));
    return optFLocalN;
}

//////////////////////////笛卡尔坐标系//////////////////////////

void EnuToEciFunc(const double *pEnuCoord, double lambda, double phi, double *pEciCoord) {
    double x = pEnuCoord[0], y = pEnuCoord[1], z = pEnuCoord[2];

    double sinL = sin(lambda), cosL = cos(lambda);
    double sinP = sin(phi),    cosP = cos(phi);

    pEciCoord[0] = z*cosL*cosP - x*sinL - y*cosL*sinP;
    pEciCoord[1] = x*cosL + z*cosP*sinL - y*sinL*sinP;
    pEciCoord[2] = y*cosP + z*sinP;
}

void CartesianStateFcn(const double *pState, const double *pControl,
                       double tm, double mu, double *pDotState) {
    double x = pState[0], y = pState[1], z = pState[2];
    double lambda = atan2(y, x), phi = atan2(z, sqrt(x*x + y*y));

    memcpy(pDotState, pState + 3, g_DOUBLE_3);

    double gDotVEnu[3] { 0, 0,
                         -mu/(x*x + y*y + z*z),
    }, gDotVEci[3];

    if (pControl != nullptr) {
        double alpha = pControl[0], beta = pControl[1];

        gDotVEnu[0] += tm*cos(beta)*cos(alpha);
        gDotVEnu[1] += tm*cos(beta)*sin(alpha);
        gDotVEnu[2] += tm*sin(beta);
    }

    EnuToEciFunc(gDotVEnu, lambda, phi, gDotVEci);

    memcpy(pDotState + 3, gDotVEci, g_DOUBLE_3);
}

void CartesianCostateFcn(const double *pState, const double *pCostate, const double *pControl,
                         double tm, double mu, double *pDotCostate) {
    double l4 = pCostate[3], l5 = pCostate[4], l6 = pCostate[5];

    double x = pState[0], y = pState[1], z = pState[2];
    double xy = x*y, xz = x*z, yz = y*z;
    double x2 = x*x, y2 = y*y, z2 = z*z;

    double r = sqrt(x2 + y2), R = sqrt(x2 + y2 + z2);
    double r3 = r*r*r, R3 = R*R*R;

    double alpha = pControl[0], beta = pControl[1];
    double sin_a = sin(alpha), cos_a = cos(alpha);
    double sin_b = sin(beta),  cos_b = cos(beta);

    double T = mu/R3 - tm*sin_b/R;
    double T3 = 3*mu/(R3*R*R) - tm*sin_b/R3;

    double D = tm*z*cos_b*sin_a/(r*R3) + tm*z*cos_b*sin_a/(r3*R);

    double P = tm*cos_b*sin_a/(r*R);
    double PR3 = tm*cos_b*sin_a/R3;
    double Pr3 = tm*cos_a*cos_b/r3;
    double Pr = tm*cos_a*cos_b/r;

    pDotCostate[0] = -l6*(xz*T3 + x*P - x*PR3*r) +
                     -l5*(xy*T3 + Pr - x2*Pr3 + xy*D) +
                     -l4*(x2*T3 - T - z*P + xy*Pr3 + x2*D);
    pDotCostate[1] = -l6*(yz*T3 + y*P - y*PR3*r) +
                     -l4*(xy*T3 - Pr + y2*Pr3 + xy*D) +
                     -l5*(y2*T3 - T - z*P - xy*Pr3 + y2*D);
    pDotCostate[2] = -l6*(z2*T3 - T - z*PR3*r) +
                     -l5*(yz*T3 - y*P + y*z2*PR3/r) +
                     -l4*(xz*T3 - x*P + x*z2*PR3/r);
    pDotCostate[3] = -pCostate[0];
    pDotCostate[4] = -pCostate[1];
    pDotCostate[5] = -pCostate[2];
}

void CartesianControlFcn(const double *pState, const double *pCostate, int flag, double *pControl) {
    double x = pState[0], y = pState[1], z = pState[2];
    double r = sqrt(x * x + y * y), R = sqrt(x * x + y * y + z * z);
    double sin_l = y / r, cos_l = x / r;
    double sin_p = z / R, cos_p = r / R;
    double l4 = pCostate[3], l5 = pCostate[4], l6 = pCostate[5];

    auto beta_fcn = [=](double alpha, double *pBeta) {
        double sin_a = sin(alpha), cos_a = cos(alpha);

        pBeta[0] = atan2(-l4*cos_l*cos_p - l5*sin_l*cos_p - l6*sin_p,
                         l4*(cos_a*sin_l + sin_a*cos_l*sin_p) + l5*(-cos_a*cos_l + sin_a*sin_l*sin_p) - l6*sin_a*cos_p);
        pBeta[1] = pBeta[0] > 0 ? pBeta[0] - M_PI : pBeta[0] + M_PI;
    };

    auto check_fcn = [=](double alpha, double beta) {
        double sin_a = sin(alpha), cos_a = cos(alpha);
        double sin_b = sin(beta),  cos_b = cos(beta);

        double h11 = l4*(cos_a*cos_b*sin_l + sin_a*cos_b*cos_l*sin_p) +
                     l5*(-cos_a*cos_b*cos_l + sin_a*cos_b*sin_l*sin_p) +
                     l6*(-sin_a*cos_b*cos_p);
        double h22 = l4*(cos_a*cos_b*sin_l + sin_a*cos_b*cos_l*sin_p - sin_b*cos_l*cos_p) +
                     l5*(-cos_a*cos_b*cos_l + sin_a*cos_b*sin_l*sin_p - sin_b*sin_l*cos_p) +
                     l6*(-sin_a*cos_b*cos_p - sin_b*sin_p);
        double h12 = l4*(-sin_a*sin_b*sin_l + cos_a*sin_b*cos_l*sin_p) +
                     l5*(sin_a*sin_b*cos_l + cos_a*sin_b*sin_l*sin_p) +
                     l6*(-cos_a*sin_b*cos_p);
        double h21 = h12;

        return flag*(h11 + h22) < 0 && h11*h22 - h12*h21 > 0;
    };

    double gAlpha[2];
    gAlpha[0] = atan2(l4 * cos_l * sin_p + l5 * sin_l * sin_p - l6 * cos_p,
                      l4 * sin_l - l5 * cos_l);
    gAlpha[1] = gAlpha[0] > 0 ? gAlpha[0] - M_PI : gAlpha[0] + M_PI;

    for (auto alpha : gAlpha) {
        pControl[0] = alpha;
        double gBeta[2];
        beta_fcn(alpha, gBeta);
        for (auto beta : gBeta) {
            if (check_fcn(alpha, beta)) {
                pControl[1] = beta;
                return;
            }
        }
    }
}

double CartesianHamiltonFcn(const double *pState, const double *pCostate, const double *pControl,
                            double tm, double mu) {
    double gDotState[6];
    CartesianStateFcn(pState, pControl, tm, mu, gDotState);

    double result = .0;
    for (int i = 0; i < 6; i++) result += pCostate[i]*gDotState[i];
    return result;
}

void CartesianExtStateFcn(const double *pExtState, double tmP, double tmE, double mu, double *pDotExtState) {
    const double *pStateP   = pExtState;
    const double *pStateE   = pExtState + 6;
    const double *pCostateP = pExtState + 12;
    const double *pCostateE = pExtState + 18;

    double gControlP[2], gControlE[2];
    CartesianControlFcn(pStateP, pCostateP, -1, gControlP);
    CartesianControlFcn(pStateE, pCostateE, 1,  gControlE);

    CartesianStateFcn(pStateP, gControlP, tmP, mu, pDotExtState);
    CartesianStateFcn(pStateE, gControlE, tmE, mu, pDotExtState + 6);
    CartesianCostateFcn(pStateP, pCostateP, gControlP, tmP, mu, pDotExtState + 12);
    CartesianCostateFcn(pStateE, pCostateE, gControlE, tmE, mu, pDotExtState + 18);
}

void CartesianBoundaryFcn(const double *pExtState, double tmP, double tmE, double mu, double *pBoundary) {
    const double *pStateP   = pExtState;
    const double *pStateE   = pExtState + 6;
    const double *pCostateP = pExtState + 12;
    const double *pCostateE = pExtState + 18;

    for (int i = 0; i < 3; i++) {
        pBoundary[i]     = pStateP[i]   - pStateE[i];
        pBoundary[i + 3] = pCostateP[i] + pCostateE[i];
        pBoundary[i + 6] = pCostateP[i + 3];
        pBoundary[i + 9] = pCostateE[i + 3];
    }

    double gControlP[2], gControlE[2];
    CartesianControlFcn(pStateP, pCostateP, -1, gControlP);
    CartesianControlFcn(pStateE, pCostateE, 1,  gControlE);
    pBoundary[12] = 1 + CartesianHamiltonFcn(pStateP, pCostateP, gControlP, tmP, mu) +
                    CartesianHamiltonFcn(pStateE, pCostateE, gControlE, tmE, mu);
}

double CartesianFitnessFcn(const double *pIndividual, const double *pK, const double *pStateP0, const double *pStateE0,
                           double tmP, double tmE, double du, double tu) {
    static auto norm_fcn = [](const double *pExtState, double du, double tu, double *pNormExtState) {
        for (int i = 0; i < 3; i++) {
            double vu = du/tu;
            pNormExtState[i]     = pExtState[i]/du;
            pNormExtState[6 + i] = pExtState[6 + i]/du;
            pNormExtState[3 + i] = pExtState[3 + i]/vu;
            pNormExtState[9 + i] = pExtState[9 + i]/vu;
        }
        memcpy(pNormExtState + 12, pExtState + 12, g_DOUBLE_12);
    };

    static auto denorm_fcn = [](const double *pExtState, double du, double tu, double *pDenormExtState) {
        for (int i = 0; i < 3; i++) {
            double vu = du/tu;
            pDenormExtState[i]     = pExtState[i]*du;
            pDenormExtState[6 + i] = pExtState[6 + i]*du;
            pDenormExtState[3 + i] = pExtState[3 + i]*vu;
            pDenormExtState[9 + i] = pExtState[9 + i]*vu;
        }
        memcpy(pDenormExtState + 12, pExtState + 12, g_DOUBLE_12);
    };

    double gu = du/tu/tu;
    double mu = du*du*du/tu/tu;
    double param[2] = { tmP/gu, tmE/gu };

    gsl_odeiv2_system normOdeSystem = {
            [](double t, const double y[], double f[], void *params) -> int {
                (void)(t);
                double normTmP = *((double*)params);
                double normTmE = *((double*)params + 1);

                CartesianExtStateFcn(y, normTmP, normTmE, 1, f);

                return GSL_SUCCESS;
            }, nullptr, 24, param};
    gsl_odeiv2_driver *normOdeDriver =
            gsl_odeiv2_driver_alloc_y_new(&normOdeSystem, gsl_odeiv2_step_rkf45, 1e-3, 1e-6, 1e-6);

    double t = .0, tf, gExtState[24], gNormExtState[24];

    tf = pIndividual[12];

    memcpy(gExtState,      pStateP0,        g_DOUBLE_6);
    memcpy(gExtState + 6,  pStateE0,        g_DOUBLE_6);
    memcpy(gExtState + 12, pIndividual,     g_DOUBLE_6);
    memcpy(gExtState + 18, pIndividual + 6, g_DOUBLE_6);
    norm_fcn(gExtState, du, tu, gNormExtState);

    int status = gsl_odeiv2_driver_apply(normOdeDriver, &t, tf, gNormExtState);
    if (status != GSL_SUCCESS) {
        printf ("error, return value=%d\n", status);
        exit(1);
    }
    else {
        denorm_fcn(gNormExtState, du, tu, gExtState);

        double gBoundary[13];
        CartesianBoundaryFcn(gExtState, tmP, tmE, mu, gBoundary);

        double target = .0;
        for (int i = 0; i < 13; i++) target += pK[i]*abs(gBoundary[i]);
        return target;
    }
}

double CartesianSolveFcn(Param *pParam, double *pOptIndividual, bool printProcess) {
    const unsigned int dim = 13;
    pParam->dim = dim;

    pParam->printProcess = printProcess;

    auto *pOptGlobal = new nlopt::opt(nlopt::algorithm::GN_ISRES, dim);
    auto *pOptLocalN = new nlopt::opt(nlopt::algorithm::LN_NELDERMEAD, dim);
    // auto *pOptLocalD = new nlopt::opt(nlopt::algorithm::LD_MMA, dim);

    auto optXGlobal = std::vector<double>(dim, 0);
    double optFGlobal = .0;

    auto optXLocalN = std::vector<double>(dim, 0);
    double optFLocalN = .0;

    // auto optXLocalD = std::vector<double>(dim, 0);
    // double optFLocalD = .0;

    for (int i = 0; i < dim; i++) {
        double temp = (pParam->lb[i] + pParam->ub[i])/2.0;
        optXGlobal[i] = temp;
        optXLocalN[i] = temp;
        // optXLocalD[i] = temp;
    }

    auto lb = std::vector<double>(pParam->lb, pParam->lb + dim);
    auto ub = std::vector<double>(pParam->ub, pParam->ub + dim);

    auto objective = [](const std::vector<double> &x, std::vector<double> &grad, void* f_data) {
        auto p = (Param*)f_data;
        auto result = CartesianFitnessFcn(x.data(), p->pK, p->pInitialStateP, p->pInitialStateE,
                                          p->tmP, p->tmE, p->du, p->tu);
        const unsigned int dim = p->dim;

        // if (!grad.empty()) {
        //     double delta = 1e-8;
        //     double tempX[dim];
        //     double f1, f2;
        //     memcpy(tempX, x.data(), dim*sizeof(double));
        //     for (int i = 0; i < dim; i++) {
        //         tempX[i] += delta/2;
        //         f1 = CartesianFitnessFcn(tempX, p->pK, p->pInitialStateP, p->pInitialStateE,
        //                                  p->tmP, p->tmE, p->du, p->tu);
        //         tempX[i] -= delta;
        //         f2 = CartesianFitnessFcn(tempX, p->pK, p->pInitialStateP, p->pInitialStateE,
        //                                  p->tmP, p->tmE, p->du, p->tu);
        //         tempX[i] += delta/2;
        //         grad[i] = (f1 - f2)/delta;
        //     }
        // }

        if (p->printProcess)
            cout << x.data()[dim - 1] << '\t' << result << endl;

        return result;
    };

    pOptGlobal->set_min_objective(objective, pParam);
    pOptGlobal->set_lower_bounds(lb);
    pOptGlobal->set_upper_bounds(ub);
    pOptGlobal->set_ftol_rel(1e-6);
    pOptGlobal->set_ftol_abs(1e-6);
    pOptGlobal->set_xtol_rel(1e-6);
    pOptGlobal->set_xtol_abs(1e-6);

    pOptLocalN->set_min_objective(objective, pParam);
    pOptLocalN->set_lower_bounds(lb);
    pOptLocalN->set_upper_bounds(ub);
    pOptLocalN->set_ftol_rel(1e-6);
    pOptLocalN->set_ftol_abs(1e-6);
    pOptLocalN->set_xtol_rel(1e-6);
    pOptLocalN->set_xtol_abs(1e-6);

    // pOptLocalD->set_min_objective(objective, pParam);
    // pOptLocalD->set_lower_bounds(lb);
    // pOptLocalD->set_upper_bounds(ub);
    // pOptLocalD->set_ftol_rel(1e-9);
    // pOptLocalD->set_ftol_abs(1e-9);
    // pOptLocalD->set_xtol_rel(1e-9);
    // pOptLocalD->set_xtol_abs(1e-9);

    pOptGlobal->optimize(optXGlobal, optFGlobal);

    optXLocalN = optXGlobal;
    optFLocalN = optFGlobal;

    pOptLocalN->optimize(optXLocalN, optFLocalN);

    // optXLocalD = optXLocalN;
    // optFLocalD = optFLocalN;

    // pOptLocalD->optimize(optXLocalD, optFLocalD);

    memcpy(pOptIndividual, optXLocalN.data(), dim*sizeof(double));
    return optFLocalN;
}
