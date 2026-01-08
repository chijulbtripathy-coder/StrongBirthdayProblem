"""
Strong Birthday Problem (SBP) Implementation 

Formula: prob(m, n, k) = (1/m^n) * sum_{j=k}^{n} (-1)^{j-k} * C(j, k) * C(m, j) * C(n, j) * j! * (m-j)^(n-j)

Where:
- m: number of days in a year
- n: number of people
- k: parameter for the probability calculation
"""

import sys 
import time 
import mpmath 
from mpmath import mp, mpf, fmul, binomial, fsub, fadd, factorial, power

# Set precision to 1000 decimal places
mp.dps = 1000

# Increase recursion limit if needed
sys.setrecursionlimit(10000)


def prob(m, n, k):
    """
    Calculate the probability for the Strong Birthday Problem 

    Args:
        m: number of days in a year
        n: number of people
        k: parameter for the probability calculation

    Returns:
        The probability as an mpmath.mpf value
    """
    # Convert to mpf for arbitrary precision
    m = mpf(m)
    n_int = int(n)
    k_int = int(k)

    # Calculate 1/m^n
    denominator = power(m, n)

    # Calculate the sum 
    sum_result = mpf(0)

    for j in range(k_int, n_int + 1):
        # Calculate each term in the sum
        # (-1)^(j-k) 
        sign = mpf(-1) ** (j - k_int)

        # C(j, k)
        binom_j_k = binomial(j, k_int)

        # C(m, j)
        binom_m_j = binomial(m, j)

        # C(n, j)
        binom_n_j = binomial(n_int, j)

        # j!
        j_factorial = factorial(j)

        # (m - j)^(n - j)
        if n_int - j > 0:
            power_term = power(fsub(m, mpf(j)), n_int - j)
        else:
            power_term = mpf(1)

        # Multiply all terms 
        term = fmul(sign, binom_j_k)
        term = fmul(term, binom_m_j)
        term = fmul(term, binom_n_j)
        term = fmul(term, j_factorial)
        term = fmul(term, power_term)

        # Add to sum 
        sum_result = fadd(sum_result, term)

    # Final result: sum / (m^n)
    result = sum_result / denominator

    return result


def run_test_cases():
    """
    Run the test cases for the Strong Birthday Problem.
    """
    test_cases = [
        (10, 41, 0),
        (50, 304, 0),
        (100, 690, 0),
        (10, 112, 0),
        (50, 665, 0),
        (100, 1410, 0)
    ]

    print("\nStrong Birthday Problem - Test Results (Formula Implementation)\n")
    print("=" * 120)
    print(f"{'Test #':<8} {'m':<8} {'n':<8} {'k':<8} {'Probability':<30} {'Time (s)':<15}")
    print("=" * 120)
    
    for idx, (m, n, k) in enumerate(test_cases, 1):
        print(f"Test {idx:<3} {m:<8} {n:<8} {k:<8}...", end=" ", flush=True)
        start_time = time.time()
        result = prob(m, n, k)
        end_time = time.time()
        elapsed = end_time - start_time
        prob_str = mpmath.nstr(result, 20, strip_zeros=False)
        print(f"{prob_str:<30} {elapsed:<15.6f}")

    print("=" * 120)


if __name__ == "__main__":
    run_test_cases()