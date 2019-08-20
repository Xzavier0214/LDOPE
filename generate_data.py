from solve_spherical_model import solve_spherical_model
from solve_cartesian_model import solve_cartesian_model

if __name__ == "__main__":

    spherical_lb = (-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, 1)
    spherical_ub = (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5)

    cartesian_lb = (-10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10,
                    2)
    cartesian_ub = (10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 2.5)

    # for _ in range(100):
    #     for case_index in range(1, 4):
    #         solve_spherical_model(case_index, spherical_lb, spherical_ub)
    #         solve_cartesian_model(case_index, cartesian_lb, cartesian_ub,
    #                               True)
    #         solve_cartesian_model(case_index, cartesian_lb, cartesian_ub,
    #                               False)

    # for _ in range(100):
    #     for case_index in range(4, 7):
    #         solve_spherical_model(case_index, spherical_lb, spherical_ub)

    # for _ in range(100):
    #     solve_spherical_model(7, spherical_lb, spherical_ub)
    #     solve_cartesian_model(4, cartesian_lb, cartesian_ub)

    for _ in range(5):
        solve_cartesian_model(2, cartesian_lb, cartesian_ub, True)
