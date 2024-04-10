import numpy as np
from typing import List


class MarkovChain:
    def __init__(self, transition_matrix: List[List[float]], initial_state: List[float]) -> None:
        """
        Initialize the Markov Chain with transition matrix and initial state.

        Args:
        - transition_matrix (list of lists): The transition matrix of the Markov Chain.
        - initial_state (list): The initial state vector.
        """
        self.P = np.array(transition_matrix)
        self.p_initial = np.array(initial_state)

    def run_tests(self, num_tests: int = 5) -> None:
        """
        Run Markov Chain tests for a specified number of iterations.

        Args:
        - num_tests (int): The number of tests to run.
        """
        p_current = self.p_initial
        for i in range(num_tests):
            p_current = np.dot(p_current, self.P)
            print(f'Після {i + 1}-го тесту:')
            for j, p in enumerate(p_current):
                print(f'p{j + 1}({i + 1}) = {p:.4f}', end=' ')
            print('\n')

    def calculate_last_state(self, num_tests: int = 5) -> np.array:
        """
        Calculate the state vector after a specified number of iterations.

        Args:
        - num_tests (int): The number of iterations.

        Returns:
        - numpy.array: The resulting state vector.
        """
        return np.dot(self.p_initial, np.linalg.matrix_power(self.P, num_tests))


# Transition matrix and initial state
P = [
    [0.4, 0.25, 0.20, 0.10, 0.05],
    [0, 0.45, 0.25, 0.20, 0.10],
    [0, 0, 0.3, 0.45, 0.25],
    [0, 0, 0, 0.35, 0.65],
    [0, 0, 0, 0, 1]
]

p = [1, 0, 0, 0, 0]

# Create and run Markov Chain tests
mc = MarkovChain(P, p)
mc.run_tests()

# Perform transition matrix power calculation and print result
print('Перевірка p(5) = p(0) * P⁵:')
for i, p in enumerate(mc.calculate_last_state()):
    print(f'p({i + 1}) = {p:.4f}', end=' ')
