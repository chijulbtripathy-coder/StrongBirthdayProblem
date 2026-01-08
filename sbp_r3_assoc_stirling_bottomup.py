"""
Strong Birthday Problem - R3 Associated Stirling Numbers (Bottom-Up DP)

This implementation uses bottom-up dynamic programming with rolling arrays 
to compute r-associated Stirling numbers of the second kind (r=2).

Recurrence: S(n,k) = k * S(n-1, k) + (n-1) * S(n-2, k-1) + (n-1) * S(n-1, k-1)

The probability is computed as: 
prob(m, n) = (1/m^n) * sum_{k=1}^{floor(n/2)} C(m, k) * k! * S(n, k)
"""

from mpmath import mp, mpf, fac, binomial, fmul, fadd, power
import time 

# Set precision to 1000 decimal places
mp.dps


def assoc_stirling_bottomup(n_target):
    """
    Compute all S(n_target, k) for k = 1 to floor(n_target/2) using bottom-up DP.

    Uses rolling arrays to save space: only keep S(n-1, *) and S(n-2, *)
    to compute S(n, *).

    Returns a dictionary with S(n_target, k) values.
    """
    if n_target == 0:
        return {0: mpf(1)}
    
    # We need to track two previous rows: n-1 and n-2
    # prev2 = S(n-2, k) for all k 
    # prev = S(n-1, k) for all k
    # curr = S(n, k) for all k

    # Initialize for n = 0 
    prev2 = {0: mpf(1)} 

    # Initialize for n = 1
    if n_target >= 1:
        prev1 = {} # S(1, k) = 0 for all k >= 1 (can't have 2 elements per partiton with only 1 element)
    
    # Build up from n = 2 to n_target
    for n in range(2, n_target + 1):
        curr = {}
        max_k = n // 2

        for k in range(1, max_k + 1):
            if n < 2*k:
                continue

            result = mpf(0)

            # Term 1: k * S(n-1, k)
            if k in prev1:
                result = fadd(result, fmul(mpf(k), prev1[k]))

            # Term 2: (n-1) * S(n-2, k-1)
            if k - 1 in prev2:
                result = fadd(result, fmul(mpf(n - 1), prev2[k - 1]))

            if result != 0:
                curr[k] = result

        # Shift: prev2 <- prev1, prev1 <- curr
        prev2 = prev1
        prev1 = curr

    return prev1 if n_target >= 2 else prev2


def prob_r3_bottomup(m, n):
    """
    Compute Strong Birthday Problem probability using Associated Stirling numbers 
    with bottom-up DP.

    Formula: prob(m,n) = (1/m^n) * sum_{k=1}^{floor(n/2)} C(m, k) * k! * S(n, k)
    """
    # Compute all S(n, k) for k = 1 to floor(n/2
    stirling_values = assoc_stirling_bottomup(n)

    # Compute the sum: sum_{k=1}^{floor(n/2)} C(m, k) * k! * S(n, k)
    total = mpf(0)
    max_k = n // 2

    for k in range(1, max_k + 1):
        if k in stirling_values:
            stirling_val = stirling_values[k]

            # C(m, k) * k! * S(n, k)
            term = binomial(m, k) * fac(k) * stirling_val
            total += term 

    # Divide by m^n to get probability
    prob = total / mpf(m) ** n

    return prob


def main():
    """Test the immplementation with standard test cases"""
    test_cases = [ 
        (10, 41, 0),
        (50, 304, 0),
        (100, 690, 0),
        (10, 112, 0),
        (50, 665, 0),
        (100, 1410, 0)
    ]

    print("Strong Birthday Problem - R3 Associated Stirling Numbers (Bottom-Up DP)")
    print("=" * 120)
    print()

    for i, (m, n, k) in enumerate(test_cases, 1):
        print(f"Test {i}: m = {m}, n = {n} k = {k}")

        start_time = time.time()
        probability = prob_r3_bottomup(m, n)
        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"Probability: {probability}")
        print(f"Time: {elapsed_time:.3f} seconds")
        print()


if __name__ == "__main__":
    main()
        