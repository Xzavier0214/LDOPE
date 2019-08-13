from spherical_solve import spherical_solve
from cartesian_solve import cartesian_solve

if __name__ == "__main__":

    spherical_lb = (-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 1)
    spherical_ub = (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5)

    cartesian_lb = (-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,
                    1)
    cartesian_ub = (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5)

    for _ in range(100):
        for case_index in range(1, 4):
            spherical_solve(case_index, spherical_lb, spherical_ub)
            cartesian_solve(case_index, cartesian_lb, cartesian_ub, True)

    # for _ in range(100):
    #     for case_index in range(1, 4):
    #         cartesian_solve(case_index, cartesian_lb, cartesian_ub, False)
