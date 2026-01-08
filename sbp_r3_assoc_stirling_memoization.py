"""
Strong Birthday Problem - R3: Associated Stirling Numbers (Memoization)

This implementation uses the r-associated Stirling numbers of the second kid 
to compute the Strong Birthday Problem probability. For r=2:

S(n, k) = k * S(n-1, k) + (n-1) * S(n-2, k-1) 

where S(n, k) = {n \brace k}_{>=2}

The probability is computed as:
prob(m, n) = (1/m^n) * sum_{k=1}^{floor(n/2)} C(m, k) * k! * S(n, k)

This counts ways to choose k days and distribute n people to those days 
with each day having at least 2 birthdays.
"""

from mpmath import mp, mpf, fac, binomial, fmul, fadd, power
import time     
import sys  

# Set precision to 1000 decimal places
mp.dps = 1000

# Increase recursion limit to handle large n 
sys.setrecursionlimit(10000)


def assoc_stirling(n, k, memo=None):
    """
    Compute {n \brace k}_{>=2} using memoization.

    This is the r-associated Stirling number of the second kind for r=2.
    counting ways to partition n items into k non-empty subsets where
    each subset has at least 2 elements.

    Recurrence: S(n, k) = k * S(n-1, k) + (n-1) * S(n-2, k-1)
    Base cases: S(0, 0) = 1; S(n, 0) = 0 if invalid  
    """
    if (n, k) in memo:
        return memo[(n, k)] 
    
    # Base cases
    if (n <= 0 and k > 0) or (n > 0 and k <= 0) or (n < 2*k):
        return mpf(0)
    
    if n == 0 and k == 0:
        return mpf(1)

    # Recurrence: S(n, k) = k * S(n-1, k) + (n-1) * S(n-2, k-1)
    term1 = fmul(mpf(k), assoc_stirling(n-1, k, memo))
    term2 = fmul(mpf(n-1), assoc_stirling(n-2, k-1, memo))

    result = fadd(term1, term2)
    memo[(n, k)] = result
    return result


def prob_r3_memo(m, n):
    """
    Compute Strong Birthday Problem probability using Associated Stirling numbers 
    with memoization

    Formula: prob(m, n) = (1/m^n) * sum_{k=1}^{floor(n/2)} C(m, k) * k! * S(n, k)
    """
    memo = {}

    # Compute the sum sum_C(m, k) * k! * S(n, k)
    total = mpf(0)
    max_k = n // 2

    for k in range(1, max_k + 1):
        # S(n, k) = {n \brace k}_{>=2}
        stirling_val = assoc_stirling(n, k, memo)
        
        if stirling_val != 0:
            # C(m, k) * k! * S(n, k)
            term = fmul(fmul(binomial(m, k), fac(k)), stirling_val)
            total = fadd(total, term)

    # Divide by m^n to get probability
    prob = total / power(mpf(m), n)

    return prob


def main():
    """Test the implementation with standard test cases"""
    test_cases = [ 
        (10, 41, 0),
        (50, 304, 0),
        (100, 690, 0),
        (10, 112, 0),
        (50, 665, 0),
        (100, 1410, 0)
    ]

    print("Strong Birthday Problem - R3 Associated Stirling Numbers (Memoization)")
    print("=" * 80)
    print()

    for i, (m, n, k) in enumerate(test_cases, 1):
        print(f"Test Case {i}: m={m}, n={n} k={k}")

        start_time = time.time()
        probability = prob_r3_memo(m, n)
        end_time = time.time()

        elapsed_time = end_time - start_time

        print(f"Probability: {probability}")
        print(f"Time: {elapsed_time:.3f} seconds")
        print()


if __name__ == "__main__":
    main()