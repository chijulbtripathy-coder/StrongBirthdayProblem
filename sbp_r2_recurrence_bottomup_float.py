"""
Strong Birthday Problem (SBP) - R2 Probability Recurrence Bottom-Up DP (Float)

This implements the probability recurrence relation using bottom-up DP (tabulation):
P(j, k, n, m) = (1/m) * [ j * P(j, k, n-1, m)
                        + (k+1) * P(j-1, k+1, n-1, m)
                        + (m-j-k+1) * P(j, k-1, n-1, m)]

Bottom-up implementation using standard Python floats with rolling arrays.
"""

import sys
import time

def P_bottomup(n_target, m):
    """
    Calculate P(j, k, n, m) for all needed (j, k) pairs up to n_target using bottom-up DP.

    Args:
        n_target: target value of n 
        m: number of days in a year

    Returns:
        Dictionary with P valyes for n=n_target
    """
    m_inv = 1.0 / m
    # Rolling arrays 
    prev = {}
    curr = {}

    # Base case for n=0
    prev[(0, 0)] = 1.0

    # Build up from n = 1 to n = n_target
    for n in range(1, n_target + 1):
        curr = {}

    
        max_j = n // 2
        max_k = min(n, m)

        for j in range(0, max_j + 1):
            for k in range(0, max_k + 1):
                # Validity check
                if n < 2*j + k:
                    continue

                result = 0.0

                # Term 1: j * P(j, k, n-1, m)
                if (j, k) in prev:
                    result += j * prev[(j, k)]

                # Term 2: (k+1) * P(j-1, k+1, n-1, m)
                if j > 0 and (j-1, k+1) in prev:
                    result += (k + 1) * prev[(j-1, k+1)]

                # Term 3: (m-j-k+1) * P(j, k-1, n-1, m)
                if k > 0 and (j, k-1) in prev:
                    result += (m-j-k+1) * prev[(j, k-1)]

                # Multiply by 1/m
                result *= m_inv

                if result != 0:
                    curr[(j, k)] = result

        prev = curr

    return curr


def prob_r2_bottomup_float(m, n):
    """
    Calculate the Probability for the Strong Birthday Problem using R2 recurrence (bottom-up DP, float).

    Args:
        m: number of days in a year 
        n: number of people 

    Returns:
        The R2 probability as a float
    """
    n_int = int(n)

    # Compute P values using bottom-up DP
    P_values = P_bottomup(n_int, m)

    # Calculate the sum: sum_{j=1}^{floor(n/2)} P(j, 0, n, m)
    result = 0.0
    max_j = n_int // 2

    for j in range(1, max_j + 1):
        if (j, 0) in P_values:
            result += P_values[(j, 0)]

    return result


def run_test_cases(): 
    """
    Run the test cases for the Strong Birthday Problem using R2 probability recurrence (bottom-up DP, float).
    """
    test_cases = [ 
        (10, 41, 0),
        (50, 304, 0),
        (100, 690, 0),
        (10, 112, 0),
        (50, 665, 0),
        (100, 1410, 0)
    ]

    print("\nStrong Birthday Problem - R2 Probability Recurrence Bottom-Up DP (Float)")
    print("=" * 120)
    print(f"{'Test #':<8} {'m':<8} {'n':<8} {'k':<8} {'Probability':<30} {'Time (s)':<15}")
    print("=" * 120)

    for idx, (m, n, k) in enumerate(test_cases, 1):
        print(f"Test {idx:<3} {m:<8} {n:<8} {k:<8}", end=" ", flush=True)
        start_time = time.time()
        result = prob_r2_bottomup_float(m, n)
        end_time = time.time()
        elapsed_time = end_time - start_time
        prob_str = f"{result:.20f}"
        print(f"{prob_str:<30} {elapsed_time:<15.6f}")

    print("=" * 120)


if __name__ == "__main__":
    run_test_cases()


  