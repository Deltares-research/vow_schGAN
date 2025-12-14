"""
Calculate the number of possible CPT selection scenarios.

Constraints:
- Total CPTs: 36 (numbered 1-36)
- Select: 12 CPTs per scenario
- Never select CPT 1 or CPT 36
- Never select two consecutive CPTs
"""

from math import comb
from itertools import combinations


def count_valid_combinations_formula():
    """
    Calculate valid combinations using the mathematical formula.

    When selecting k items from n items such that no two are consecutive,
    the formula is: C(n - k + 1, k)

    In our case:
    - Available CPTs: 2 through 35 (34 total, excluding 1 and 36)
    - Select: 12 CPTs
    - Formula: C(34 - 12 + 1, 12) = C(23, 12)
    """
    n = 34  # Available CPTs (2-35)
    k = 12  # CPTs to select

    # Formula for non-consecutive selections: C(n - k + 1, k)
    result = comb(n - k + 1, k)
    return result


def is_valid_selection(selection):
    """
    Check if a selection is valid (no consecutive CPTs).

    Args:
        selection: tuple of selected CPT indices

    Returns:
        bool: True if valid, False otherwise
    """
    for i in range(len(selection) - 1):
        if selection[i + 1] - selection[i] == 1:
            return False
    return True


def count_valid_combinations_bruteforce():
    """
    Count valid combinations by checking all possibilities.
    WARNING: This is computationally expensive and only for verification!

    Returns:
        int: Number of valid combinations
    """
    # Available CPTs: 2 through 35
    available_cpts = range(2, 36)

    # Generate all combinations of 12 CPTs
    all_combinations = combinations(available_cpts, 12)

    # Count valid combinations
    valid_count = sum(1 for combo in all_combinations if is_valid_selection(combo))

    return valid_count


def main():
    print("=" * 70)
    print("CPT Selection Scenario Calculation")
    print("=" * 70)
    print("\nConstraints:")
    print("  - Total CPTs: 36 (numbered 1-36)")
    print("  - Available for selection: CPTs 2-35 (34 CPTs)")
    print("  - CPTs to select per scenario: 12")
    print("  - No two consecutive CPTs can be selected")
    print()

    # Calculate using formula (fast)
    print("Calculating using mathematical formula...")
    formula_result = count_valid_combinations_formula()
    print(f"  Number of valid combinations: {formula_result:,}")
    print()

    # Optional: Verify with brute force (slow, commented out by default)
    verify_bruteforce = False  # Set to True to verify (takes time!)

    if verify_bruteforce:
        print("Verifying with brute-force method (this may take a while)...")
        bruteforce_result = count_valid_combinations_bruteforce()
        print(f"  Brute-force result: {bruteforce_result:,}")
        print()

        if formula_result == bruteforce_result:
            print("✓ Results match! Formula is correct.")
        else:
            print("✗ Results don't match! Check the logic.")
    else:
        print(
            "(Brute-force verification is disabled. Set verify_bruteforce=True to enable)"
        )

    print("=" * 70)
    print(f"\nFINAL ANSWER: {formula_result:,} possible scenarios")
    print("=" * 70)


if __name__ == "__main__":
    main()
