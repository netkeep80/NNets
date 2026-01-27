// Test program to verify the NaN fix for percentage calculation
// This demonstrates the bug where NaN values were not handled correctly,
// resulting in -2147483648% being displayed instead of 0%

#include <iostream>
#include <cmath>
#include <limits>

using namespace std;

// Simulates the BUGGY behavior (original code)
void display_percentage_buggy(float neural_output) {
    float z1 = neural_output * 100.0f;
    if (z1 < 0.0) z1 = 0.;
    if (z1 > 100.0) z1 = 100.0f;
    cout << "Buggy:  " << long(z1) << "%" << endl;
}

// Simulates the FIXED behavior (new code with NaN check)
void display_percentage_fixed(float neural_output) {
    float z1 = neural_output * 100.0f;
    // Handle NaN and infinite values that cause incorrect percentage display
    if (!std::isfinite(z1)) z1 = 0.0f;
    if (z1 < 0.0f) z1 = 0.0f;
    if (z1 > 100.0f) z1 = 100.0f;
    cout << "Fixed:  " << long(z1) << "%" << endl;
}

int main() {
    cout << "=== Testing NaN handling in percentage display ===" << endl;
    cout << endl;

    // Test case 1: Normal value
    cout << "Test 1: Normal value (0.75)" << endl;
    display_percentage_buggy(0.75f);
    display_percentage_fixed(0.75f);
    cout << endl;

    // Test case 2: NaN value (this is what causes -2147483648%)
    cout << "Test 2: NaN value (0.0/0.0)" << endl;
    float nan_val = 0.0f / 0.0f;
    cout << "  Raw value is NaN: " << (std::isnan(nan_val) ? "yes" : "no") << endl;
    display_percentage_buggy(nan_val);
    display_percentage_fixed(nan_val);
    cout << endl;

    // Test case 3: Positive infinity
    cout << "Test 3: Positive infinity (1.0/0.0)" << endl;
    float pos_inf = 1.0f / 0.0f;
    cout << "  Raw value is infinite: " << (std::isinf(pos_inf) ? "yes" : "no") << endl;
    display_percentage_buggy(pos_inf);
    display_percentage_fixed(pos_inf);
    cout << endl;

    // Test case 4: Negative infinity
    cout << "Test 4: Negative infinity (-1.0/0.0)" << endl;
    float neg_inf = -1.0f / 0.0f;
    cout << "  Raw value is infinite: " << (std::isinf(neg_inf) ? "yes" : "no") << endl;
    display_percentage_buggy(neg_inf);
    display_percentage_fixed(neg_inf);
    cout << endl;

    // Test case 5: Large multiplication that may overflow
    cout << "Test 5: Very large value (1e38 * 10)" << endl;
    float large_val = 1e38f * 10.0f;
    cout << "  Raw value is finite: " << (std::isfinite(large_val) ? "yes" : "no") << endl;
    display_percentage_buggy(large_val);
    display_percentage_fixed(large_val);
    cout << endl;

    cout << "=== Fix verification complete ===" << endl;
    cout << "The -2147483648% output indicates integer overflow from NaN-to-long conversion." << endl;
    cout << "The fix handles NaN/infinite values by treating them as 0%." << endl;

    return 0;
}
