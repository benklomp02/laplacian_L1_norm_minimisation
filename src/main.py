import numpy as np
import time

import src.utils.runner as runner
import src.algorithms.greedy as greedy
import src.utils.objectives as obj
from src.utils.errors import *


def evaluate_objective():
    objective_class_list = [
        obj.Original,
        obj.MaxWDirection,
        obj.MinWDirection,
        obj.AbsDirectionDiff,
        obj.MaxWeightDirection,
        obj.MinWeightDirection,
        obj.RandomObjective,
    ]
    basis_list = [greedy.compute_basis] * len(objective_class_list)
    for error_strategy in [
        # TODO: Uncomment the error strategies as needed
        # RelativeProjectionNormGain,
        # GraphSmoothnessError,
        # DiagonalizationError,
        L1Variation,
    ]:
        runner.run_experiment(
            range(10, 100, 5),
            basis_list,
            objective_class_list,
            error_strategy=error_strategy,
        )


def main():
    print("Running experiments...")
    start_time = time.time()
    evaluate_objective()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
