"""
Strong Birthday Problem (SBP) - R2 Probability Recurrence Implementation (Memoization Standard Float)

This implements the probability recurrence relation directly using standard Python floats:
P(j, k, n, m) = (1/m) * [ j * P(j, k, n-1, m)
                        + (k+1) * P(j-1, k+1, n-1, m)
                        + (m-j-k+1) * P(j, k-1, n-1, m)]

Where:
- j: number of days with 2 or more birthdays
- k: number of days with singleton birthdays
- n: number of people
- m: number of days in a year
- P(j, k, n, m): probability (not count)

For SBP, we set k=0 and compute:
prob(m, n) = sum_{j=1}^{floor(n/2)} P(j, 0, n, m)

This version uses standard Python floats for speed instead of mpmath for arbitrary precision.
"""

import sys
import time


# Increase recursion limit if needed
sys.setrecursionlimit(10000)


def P(j, k, n, m, memo=None):
    """
    Calculate P(j, k, n, m) using dynamic programming with memoization.

    P(j, k, n, m) represents the probability that n birthdays are distributed
    over m possible days where:
    - j days have 2 or more birthdays
    - k days have singleton birthdays
    - m - j - k days are birthday-free

    Args:
        j: number of days with 2 or more birthdays
        k: number of days with singleton birthdays
        n: number of people
        m: number of days in a year
        memo: memoization dictionary

    Returns:
        The probability as a float
    """
    if memo is None:
        memo = {}

    # Create a key for memoization
    key = (j, k, n, m)
    if key in memo:
        return memo[key]

    # Base cases
    # Case 1: Invalid states
    if j < 0 or k < 0 or n < 2*j + k or m < j + k:
        return 0.0
    # Case 2: Empty case
    elif j == 0 and k == 0 and n == 0:
        return 1.0
    # Case 3: Recurrence relation for probability
    else:
        m_inv = 1.0 / m # 1/m factor 

        # Term 1: j * P(j, k, n-1, m)
        term1 = j * P(j, k, n-1, m, memo)

        # Term 2: (k+1) * P(j-1, k+1, n-1, m)
        term2 = (k + 1) * P(j-1, k+1, n-1, m, memo)

        # Term 3: (m-j-k+1) * P(j, k-1, n-1, m)
        term3 = (m-j-k+1) * P(j, k-1, n-1, m, memo)

        # Sum all terms and multiply by 1/m
        sum_terms = term1 + term2 + term3
        result = m_inv * sum_terms

    memo[key] = result
    return result


def prob_r2_float(m, n):
    """
    Calculate the probability for the Strong Birthday Problem using R2 probability recurrence.

    For SBP, k=0 and we compute:
    prob(m, n) = sum_{j=1}^{floor(n/2)} P(j, 0, n, m)

    Args:
        n: number of people
        m: number of days in a year

    Returns:
        The probability as a float
    """
    # Create memoization dictionary 
    memo = {}

    # Calculate the sum: sum_{j=1}^{floor(n/2)} P(j, 0, n, m)
    result = 0.0
    max_j = n // 2

    for j in range(1, max_j + 1):
        term = P(j, 0, n, m, memo)
        result += term

    return result


def run_test_cases():
    """
    Run the test cases for the Strong Birthday Problem using R2 probability recurrence ( float).
    """
    test_cases = [ 
        (10, 41, 0),
        (50, 304, 0),
        (100, 690, 0),
        (10, 112, 0),
        (50, 665, 0),
        (100, 1410, 0)
    ]

    print("\nStrong Birthday Problem - R2 Probability Recurrence Test Results (Memoization, Standard Float)")
    print("=" * 120)
    print(f"{'Test #':<8} {'m':<8} {'n':<8} {'k':<8} {'Probability':<30} {'Time (s)':<15}")
    print("=" * 120)

    for idx, (m, n, k) in enumerate(test_cases, 1):
        print(f"Test {idx:<3} {m:<8} {n:<8} {k:<8}", end=" ", flush=True)
        start_time = time.time()
        result = prob_r2_float(m, n)
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Format to 20 decimal places for comparison
        prob_str = f"{result:.20f}"
        print(f"{prob_str:<30} {elapsed_time:<15.6f}")

    print("=" * 120)


if __name__ == "__main__":
    run_test_cases()

