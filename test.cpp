#include<iostream>
#include<math.h>
#include"core.h"
using namespace std;

int main() {

    // 地球半径
    const double RE = 6378.137e3;

    // 地球引力常数
    const double MU = 3.986004418e14;

    // 归一化常数
    const double DU = RE;
    const double TU = sqrt(DU*DU*DU / MU);

    // 标准速度常量
    const double VU = DU / TU;

    // 标准加速度常量
    const double GU = DU / (TU*TU);

    // 角度与弧度转换常数
    const double DEG_TO_RAD = M_PI / 180.0;

    double tmP = 0.1*GU, tmE = 0.05*GU;

    double gSphereP[6] = { 6578.165e3, 7.784e3, 0, DEG_TO_RAD*20.0, 0,              DEG_TO_RAD*30.0 };
    double gSphereE[6] = { 6578.165e3, 7.784e3, 0, DEG_TO_RAD*46.5, DEG_TO_RAD*7.6, DEG_TO_RAD*49.6 };

    double gCostate[6] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    double gK[13] {1, DU, DU, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    double gLb[11] {-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 0};
    double gUb[11] {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};

    Param param = {
            gSphereP, gSphereE, tmP, tmE, gLb, gUb, gK, DU, TU
    };

    param.printProcess = true;

    double gOptIndividual[11], optFitness;
    SphericalSolveFcn(&param, gOptIndividual, optFitness);
    cout << optFitness << endl;

    return 0;
}