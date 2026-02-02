// =============================================================================
// QEDMMA-Lite v3.0 - Zero-DSP Correlator HLS Testbench
// =============================================================================
// Copyright (C) 2026 Dr. Mladen Mešter / Nexellum
// License: AGPL-3.0-or-later
// =============================================================================

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ap_int.h>

// Include DUT header
#define N_SAMPLES 64

typedef ap_uint<N_SAMPLES> packed_bits_t;
typedef ap_int<12> corr_t;

// Function prototype (from zero_dsp_correlator.cpp)
corr_t correlate_single(packed_bits_t x_ref, packed_bits_t y_shifted);

// Reference implementation
int correlate_reference(unsigned long long x, unsigned long long y, int n_bits) {
    unsigned long long xor_result = x ^ y;
    int hamming = __builtin_popcountll(xor_result);
    return n_bits - 2 * hamming;
}

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "Zero-DSP Correlator HLS Testbench" << std::endl;
    std::cout << "============================================" << std::endl;
    
    int errors = 0;
    const int N_TESTS = 100;
    
    srand(42);
    
    for (int t = 0; t < N_TESTS; t++) {
        // Generate random test vectors
        packed_bits_t x_ref = 0;
        packed_bits_t y_test = 0;
        
        for (int i = 0; i < N_SAMPLES; i++) {
            if (rand() % 2) x_ref[i] = 1;
            if (rand() % 2) y_test[i] = 1;
        }
        
        // Run DUT
        corr_t dut_result = correlate_single(x_ref, y_test);
        
        // Run reference
        int ref_result = correlate_reference(
            x_ref.to_uint64(), 
            y_test.to_uint64(), 
            N_SAMPLES
        );
        
        // Compare
        if (dut_result != ref_result) {
            std::cout << "ERROR at test " << t << ": ";
            std::cout << "DUT=" << dut_result << ", REF=" << ref_result << std::endl;
            errors++;
        }
    }
    
    // Test correlation peak detection
    std::cout << std::endl;
    std::cout << "Testing correlation peak detection..." << std::endl;
    
    // Generate PRN-like sequence
    packed_bits_t prn = 0;
    for (int i = 0; i < N_SAMPLES; i++) {
        prn[i] = (rand() % 2);
    }
    
    // Shift and correlate
    corr_t max_corr = -128;
    int peak_lag = -1;
    
    for (int lag = 0; lag < 32; lag++) {
        packed_bits_t shifted = (prn >> lag) | (prn << (N_SAMPLES - lag));
        corr_t corr = correlate_single(prn, shifted);
        
        if (corr > max_corr) {
            max_corr = corr;
            peak_lag = lag;
        }
    }
    
    std::cout << "Peak correlation: " << max_corr << " at lag " << peak_lag << std::endl;
    
    if (peak_lag != 0) {
        std::cout << "WARNING: Expected peak at lag 0 for autocorrelation!" << std::endl;
        errors++;
    }
    
    // Final result
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    if (errors == 0) {
        std::cout << "✅ ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << "❌ " << errors << " TESTS FAILED!" << std::endl;
    }
    std::cout << "============================================" << std::endl;
    
    return errors;
}
