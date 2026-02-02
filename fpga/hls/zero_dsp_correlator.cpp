
// QEDMMA-Lite v3.0 - Zero-DSP Correlator (HLS)
// Copyright (C) 2026 Dr. Mladen Mešter / Nexellum
// License: AGPL-3.0-or-later
// For commercial licensing: mladen@nexellum.com

#include <ap_int.h>
#include <hls_stream.h>

#define N_SAMPLES 1024
#define N_LAGS 256

// Packed bit type for parallel XOR
typedef ap_uint<N_SAMPLES> packed_bits_t;
typedef ap_int<12> corr_t;  // Correlation output

// Popcount using parallel tree reduction
template<int W>
ap_uint<clog2(W)+1> popcount(ap_uint<W> x) {
    #pragma HLS INLINE
    ap_uint<clog2(W)+1> count = 0;
    for (int i = 0; i < W; i++) {
        #pragma HLS UNROLL
        count += x[i];
    }
    return count;
}

// Single-lag correlator
corr_t correlate_single(
    packed_bits_t x_ref,
    packed_bits_t y_shifted
) {
    #pragma HLS INLINE
    #pragma HLS PIPELINE II=1
    
    packed_bits_t xor_result = x_ref ^ y_shifted;
    ap_uint<11> hamming = popcount<N_SAMPLES>(xor_result);
    
    // R = N - 2*hamming
    return (corr_t)(N_SAMPLES - 2 * hamming);
}

// Multi-lag correlator (fully pipelined)
void zero_dsp_correlator(
    packed_bits_t x_ref,
    hls::stream<ap_uint<1>>& y_stream,
    hls::stream<corr_t>& corr_out
) {
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS INTERFACE axis port=y_stream
    #pragma HLS INTERFACE axis port=corr_out
    
    static packed_bits_t y_shift_reg = 0;
    
    // Shift in new sample
    ap_uint<1> y_new = y_stream.read();
    y_shift_reg = (y_shift_reg << 1) | y_new;
    
    // Compute correlation
    corr_t r = correlate_single(x_ref, y_shift_reg);
    corr_out.write(r);
}
