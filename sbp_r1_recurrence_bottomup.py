"""
Strong Birthday Problem (SBP) - R1 Reccurence Bottom-Up Dynamic Programming 

This Implements the recurrence relation using bottom-up DP (tabulation):
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

This bottom up implementation is optimized for time efficiency using: 
- Tabulation instead of recursion 
- Rolling arrays to save space and improve cache locality 
- Pre-allocated arrays 
- Only computing necessary states 
"""

import sys 
import time 
import mpmath 
from mpmath import mp, mpf, fmul, fadd, power

# Set precision to 1000 decimal places 
mp.dps = 1000


def T_bottomup(n_target, m):
    """
    Calculate T(j, k, n, m) for all needed (j, k) pairs up to n_target using bottom up DP.

    Args: 
        n_target: target value of n 
        m: number of days in a year 

    Returns: 
        Dictionary with T values for n=n_target 
    """
    m_val = mpf(m)
    m_int = int(m)

    # We use rolling arrays: only need current and previous n 
    # We use dictionaries for sparse storage since most states are 0 
    # prev[j, k] = T(j, k, n-1, m)
    # curr[j, k] = T(j, k, n, m)

    prev = {}
    curr = {}

    # Base case: n = 0 
    # T(0, 0, 0, m) = 1, all others are 0 
    prev[(0, 0)] = mpf(1)

    # Build up from n = 1 to n = n_target 
    for n in range (1, n_target + 1):
        curr = {}

        # For each n, we need to compute T(j, k, n, m) for valid (j, k)
        # Valid ranges: 
        # - j >= 0, k >= 0
        # - n >= 2*j + k (otherwise T = 0)
        # - m >= j + k (otherwise T = 0)
        # - For SBP, we ultimately only need k = 0 but intermediate computations may need other k values

        # We need tp be careful about which k values to compute
        # From the recurrence, if we want T(j, k, n, m), we need:
        # - T(j, k, n-1, m) 
        # - T(j-1, k+1, n-1, m)
        # - T(j, k-1, n-1, m)

        # So k can go up to n (theoretically)
        # But we can optimize: for SBP we only need T(j, 0, n, m) at the end
        # Working backwards: to compute T(j, 0, n, m), we might need T(j, 1, n-1, m) 
        # To compyter T(j, k-1, n-1, m), we might need T(j-2, 2, n-1, m), etc.

        max_k = min(n, m_int) # Practical upper bound 
        max_j = n // 2 # Since n>= 2*j when k=0, and k>=0

        for j in range(0, max_j + 1):
            for k in range(0, max_k + 1):
                # Check validity: n >= 2*j + k and m >= j + k
                if n < 2*j + k or m_int < j + k:
                    continue

                # Apply recurrence:
                result = mpf(0)

                # Term 1: j * T(j, k, n-1, m)
                if (j, k) in prev:
                    term1 = fmul(mpf(j), prev[(j, k)])
                    result = fadd(result, term1)
                
                # Term 2: (k+1) * T(j-1, k+1, n-1, m)
                if j > 0 and (j-1, k+1) in prev:
                    term2 = fmul(mpf(k+1), prev[(j-1, k+1)])
                    result = fadd(result, term2)

                # Term 3: (m-j-k+1) * T(j, k-1, n-1, m)
                if k > 0 and (j, k-1) in prev:
                    term3 = fmul(mpf(m_int - j - k + 1), prev[(j, k-1)])
                    result = fadd(result, term3)

                # Only store non-zero values to save space 
                if result != 0:
                    curr[(j, k)] = result

        # Swap arrays for next interation 
        prev = curr 

    return curr  


def prob_r1_bottomup(m, n):
    """
    Calculate the probability for the Strong Birthday Problem using R1 recurrence (bottom-up DP).
    
    For SBP, k=0 and we compute:
    prob(m, n) = (1/m^n) * sum_{j=1}^{floor(n/2)} T(j, 0, n, m)

    Args:
        m: number of days in a year
        n: number of people

    Returns:
        The probability as an mpmath.mpf float.
    """
    m_val = mpf(m)
    n_int = int(n)

    # Compute T values using bottom up DP 
    T_values = T_bottomup(n_int, m_val)

    # Calculate the sum: sum_{j=1}^{floor(n/2)} T(j, 0, n, m)
    N = mpf(0)
    max_j = n_int // 2

    for j in range(1, max_j + 1):
        if (j, 0) in T_values:
            N = fadd(N, T_values[(j, 0)])

    # Calculate the probability: (1/m^n) * N
    denominator = power(m_val, n_int)
    result = N / denominator

    return result


def run_test_cases():
    """
    Run the test cases for the Strong Birthday Problem using R1 recurrence (bottom-up DP).
    """
    test_cases = [
        (10, 41, 0),
        (50, 304, 0),
        (100, 690, 0),
        (10, 112, 0),
        (50, 665, 0),
        (100, 1410, 0)
    ]

    print("\nStrong Birthday Problem - R1 Recurrence Bottom-Up DP Test Results\n")
    print("=" * 120)
    print(f"{'Test #':<8} {'m':<8} {'n':<8} {'k':<8} {'Probability':<30} {'Time (s)':<15}")
    print("=" * 120)
    
    for idx, (m, n, k) in enumerate(test_cases, 1):
        print(f"Test {idx:<3} {m:<8} {n:<8} {k:<8}...", end=" ", flush=True)
        start_time = time.time()
        result = prob_r1_bottomup(m, n)
        end_time = time.time()
        elapsed = end_time - start_time
        prob_str = mpmath.nstr(result, 20, strip_zeros=False)
        print(f"{prob_str:<30} {elapsed:<15.6f}")

    print("=" * 120)


if __name__ == "__main__":
    run_test_cases()