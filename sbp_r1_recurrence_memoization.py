"""
Strong Birthday Problem (SBP) - R1 Recurrence Implementation

This implements the recurrence relation:
T(j, k, n, m) = j * T(j, k, n-1, m) 
              + (k+1) * T(j-1, k+1, n-1, m) 
              + (m-j-k+1) * T(j, k-1, n-1, m)

Where:
- j: number of days with 2 or more birthdays 
- k: number of days with singleton birthdays 
- n: number of people 
- m: number of days in a year

For SBP, we set k=0 and compute: 
prob(m, n) = (1/m^n) * sum_{j=1}^{floor(n/2)} T(j, 0, n, m)
"""

import sys 
import time 
import mpmath 
from mpmath import mp, mpf, fmul, fadd, power

# Set precision to 1000 dwcimal places 
mp.dps = 1000

# Increase recursion limit if needed 
sys.setrecursionlimit(10000)


def T(j, k, n, m, memo=None):
    """
    Calculate T(j, k, n, m) using dynamic programming with memoization.

    T(j, k, n, m) represents the number of ways n birthdays are distrubted 
    over m possible days where:
    - j days have 2 or more birthdays
    - k days have singleton birthdays
    - m - j - k days are birthday-free

    Args:
        j: number of days with 2 or more birthdays 
        k: number of days with singleton birthdays 
        n: number of people 
        m: number of days in a year 
        memo: dictionary for memoization

    Returns:
        The count as an mpmath.mpf value 
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
        return mpf(0)
    # Case 2: Empty assignment
    elif j == 0 and k == 0 and n == 0:
        return mpf(1)
    # Case 3: Recurrence relation 
    else:
        # Term 1: Add to days with 2 or more birthdays
        term1 = fmul(mpf(j), T(j, k, n-1, m, memo))

        # Term 2: Add to singleton birthdays
        term2 = fmul(mpf(k+1), T(j-1, k+1, n-1, m, memo))

        # Term 3: Create a singleton birthday
        term3 = fmul(mpf(m - j - k + 1), T(j, k-1, n-1, m, memo))   

        # Sum all terms 
        result = fadd(fadd(term1, term2), term3)

    memo[key] = result
    return result


def prob_r1(m, n):
    """
    Calculate the probability for the Strong Birthday Problem using R1 recurrence.

    For SBP, k=0 and we compute:
    prob(m, n) = (1/m^n) * sum_{j=1}^{floor(n/2)} T(j, 0, n, m)

    Args:
        m: number of days in a year
        n: number of people

    Returns:
        The probability as an mpmath.mpf float.
    """
    # Convert to mpf for arbitrary precision
    m = mpf(m)
    n_int = int(n)

    # Create memoization dictionary
    memo = {}

    # Calculate the sum: sum_{j=1}^{floor(n/2)} T(j, 0, n, m)
    N = mpf(0)
    max_j = n_int // 2

    for j in range(1, max_j + 1):
        term = T(j, 0, n_int, m, memo)
        N = fadd(N, term)

    # Calculate probability: (1/m^n) * N
    denominator = power(m, n_int)
    result = N / denominator

    return result


def run_test_cases():
    """
    Run the test cases for the Strong Birthday Problem using R1 recurrence.
    """
    test_cases = [ 
        (10, 41, 0),
        (50, 304, 0),
        (100, 690, 0),
        (10, 112, 0),
        (50, 665, 0),
        (100, 1410, 0)
    ]

    print("\nStrong Birthday Problem - R1 Recurrence Test Results Memoization")
    print("=" * 120)
    print(f"{'Test #':<8} {'m':<8} {'n':<8} {'k':<8} {'Probability':<30} {'Time (s)':<15}")
    print("=" * 120)

    for idx, (m, n, k) in enumerate(test_cases, 1):
        print(f"Test {idx:<3} {m:<8} {n:<8} {k:<8}", end=" ", flush=True)
        start_time = time.time()
        result = prob_r1(m, n)
        end_time = time.time()
        elapsed_time = end_time - start_time
        prob_str = mpmath.nstr(result, 20, strip_zeros=False)
        print(f"{prob_str:<30} {elapsed_time:<15.6f}")

    print("=" * 120)


if __name__ == "__main__":
    run_test_cases()
