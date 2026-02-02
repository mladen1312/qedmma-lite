#!/usr/bin/env python3
"""
QEDMMA-Lite Test Suite - Zero-DSP Module
=========================================
Author:  Dr. Mladen Mešter / Nexellum d.o.o.
License: AGPL-3.0-or-later
Contact: mladen@nexellum.com | +385 99 737 5100

Run with: pytest python/tests/test_zero_dsp.py -v
"""

import numpy as np
import pytest

# Import the modules to test
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "qedmma_lite"))

from zero_dsp import (
    CSDEncoder,
    TernaryEncoder,
    RadarCodes,
    ZeroDspCorrelatorSW,
    ZeroDspCorrelator,
    benchmark_csd_vs_multiply,
)


class TestCSDEncoder:
    """Test suite for Canonical Signed Digit encoder."""

    def test_encode_simple(self):
        """Test CSD encoding of simple values."""
        # 23 = 32 - 8 - 1 = 2^5 - 2^3 - 2^0
        csd = CSDEncoder.encode(23)
        assert (0, -1) in csd  # -2^0
        assert (3, -1) in csd  # -2^3
        assert (5, 1) in csd   # +2^5

    def test_encode_power_of_two_minus_one(self):
        """Test CSD encoding of 2^n - 1 values."""
        # 127 = 128 - 1 = 2^7 - 2^0
        csd = CSDEncoder.encode(127)
        assert len(csd) == 2  # Only 2 operations
        assert (0, -1) in csd
        assert (7, 1) in csd

    def test_encode_power_of_two(self):
        """Test CSD encoding of exact powers of 2."""
        # 64 = 2^6 (single shift)
        csd = CSDEncoder.encode(64)
        assert len(csd) == 1
        assert csd[0] == (6, 1)

    def test_encode_zero(self):
        """Test CSD encoding of zero."""
        csd = CSDEncoder.encode(0)
        assert csd == []

    def test_encode_one(self):
        """Test CSD encoding of one."""
        csd = CSDEncoder.encode(1)
        assert csd == [(0, 1)]

    def test_to_operations(self):
        """Test conversion to operation string."""
        csd = CSDEncoder.encode(23)
        ops = CSDEncoder.to_operations(csd)
        # Should contain shifts and x
        assert "<<" in ops or "x" in ops

    def test_operation_count(self):
        """Test operation count calculation."""
        # 127 = 2^7 - 1 should be 2 operations
        assert CSDEncoder.operation_count(127) == 2
        # 255 = 2^8 - 1 should be 2 operations
        assert CSDEncoder.operation_count(255) == 2
        # 1 should be 1 operation
        assert CSDEncoder.operation_count(1) == 1

    def test_csd_correctness(self):
        """Verify CSD decomposition gives correct value."""
        for val in [1, 7, 15, 23, 42, 100, 127, 255, 1000]:
            csd = CSDEncoder.encode(val)
            # Reconstruct value from CSD
            reconstructed = sum(sign * (1 << pos) for pos, sign in csd)
            assert reconstructed == val, f"CSD failed for {val}"


class TestTernaryEncoder:
    """Test suite for ternary coefficient encoder."""

    def test_encode_decode_roundtrip(self):
        """Test that encode/decode is lossless."""
        coefs = np.array([1, -1, 0, 1, -1, -1, 0, 0])
        encoded = TernaryEncoder.encode(coefs)
        decoded = TernaryEncoder.decode(encoded, len(coefs))
        assert np.array_equal(coefs, decoded)

    def test_encode_all_ones(self):
        """Test encoding all +1 coefficients."""
        coefs = np.ones(8, dtype=np.int8)
        encoded = TernaryEncoder.encode(coefs)
        # Each +1 is encoded as 0b01
        assert encoded == 0x5555  # 0101_0101_0101_0101

    def test_encode_all_minus_ones(self):
        """Test encoding all -1 coefficients."""
        coefs = -np.ones(8, dtype=np.int8)
        encoded = TernaryEncoder.encode(coefs)
        # Each -1 is encoded as 0b11
        assert encoded == 0xFFFF  # 1111_1111_1111_1111

    def test_encode_all_zeros(self):
        """Test encoding all zero coefficients."""
        coefs = np.zeros(8, dtype=np.int8)
        encoded = TernaryEncoder.encode(coefs)
        assert encoded == 0

    def test_invalid_coefficients(self):
        """Test that invalid coefficients raise error."""
        coefs = np.array([1, 2, 0])  # 2 is invalid
        with pytest.raises(AssertionError):
            TernaryEncoder.encode(coefs)


class TestRadarCodes:
    """Test suite for radar waveform codes."""

    @pytest.mark.parametrize("length", [2, 3, 4, 5, 7, 11, 13])
    def test_barker_length(self, length):
        """Test Barker codes have correct length."""
        if length == 4:
            code = RadarCodes.BARKER_4A
        else:
            code = RadarCodes.get_barker(length)
        assert len(code) == length

    def test_barker_values(self):
        """Test Barker codes contain only ±1."""
        for length in [2, 3, 5, 7, 11, 13]:
            code = RadarCodes.get_barker(length)
            assert np.all(np.isin(code, [-1, 1]))

    def test_barker_13(self):
        """Test Barker-13 specific pattern."""
        b13 = RadarCodes.BARKER_13
        expected = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
        assert np.array_equal(b13, expected)

    def test_invalid_barker_length(self):
        """Test that invalid Barker length raises error."""
        with pytest.raises(ValueError):
            RadarCodes.get_barker(6)  # No Barker-6 exists

    def test_mls_generation(self):
        """Test MLS sequence generation."""
        mls = RadarCodes.mls(31)  # 2^5 - 1
        assert len(mls) == 31
        assert np.all(np.isin(mls, [-1, 1]))


class TestZeroDspCorrelatorSW:
    """Test suite for software Zero-DSP correlator."""

    def test_initialization(self):
        """Test correlator initialization."""
        corr = ZeroDspCorrelatorSW(length=64, data_width=16)
        assert corr.length == 64
        assert corr.data_width == 16

    def test_set_coefficients(self):
        """Test coefficient setting."""
        corr = ZeroDspCorrelatorSW(length=64)
        coefs = RadarCodes.BARKER_13
        corr.set_coefficients(coefs)
        assert np.array_equal(corr.coefficients[:13], coefs)

    def test_reset(self):
        """Test correlator reset."""
        corr = ZeroDspCorrelatorSW(length=64)
        corr.set_coefficients(RadarCodes.BARKER_13)
        
        # Process some samples
        for i in range(100):
            corr.process_sample(i * 100)
        
        # Reset
        corr.reset()
        
        # Delay line should be zero
        assert np.all(corr.delay_line == 0)
        assert corr._sample_count == 0

    def test_barker_correlation_peak(self):
        """Test that Barker-13 gives strong correlation peak."""
        corr = ZeroDspCorrelatorSW(length=64)
        corr.set_coefficients(RadarCodes.BARKER_13)
        
        # Create signal with Barker pulse
        signal = np.zeros(100, dtype=np.int16)
        amplitude = 1000
        signal[50:63] = (RadarCodes.BARKER_13 * amplitude).astype(np.int16)
        
        # Process
        output = corr.process(signal)
        
        # Peak should be 13 * amplitude = 13000
        expected_peak = 13 * amplitude
        assert corr.peak_value >= expected_peak * 0.9  # Allow 10% tolerance
        assert 55 <= corr.peak_index <= 70  # Peak near expected location

    def test_zero_dsp_equivalence(self):
        """Test that Zero-DSP gives same result as multiplication."""
        corr = ZeroDspCorrelatorSW(length=16)
        coefs = np.array([1, -1, 1, 1, -1, 0, 1, -1, 0, 0, 1, -1, 1, 1, -1, 1], dtype=np.int8)
        corr.set_coefficients(coefs)
        
        # Random samples
        np.random.seed(42)
        samples = np.random.randint(-1000, 1000, 50).astype(np.int16)
        
        # Process with Zero-DSP correlator
        zero_dsp_output = corr.process(samples)
        
        # Compute reference using multiplication
        corr.reset()
        ref_output = []
        delay_line = np.zeros(16, dtype=np.int16)
        for sample in samples:
            delay_line = np.roll(delay_line, 1)
            delay_line[0] = sample
            ref = np.sum(delay_line * coefs)
            ref_output.append(ref)
            corr.process_sample(sample)
        
        # Compare (should be bit-exact)
        assert np.array_equal(zero_dsp_output, np.array(ref_output))


class TestZeroDspCorrelator:
    """Test suite for unified Zero-DSP correlator interface."""

    def test_software_mode(self):
        """Test software mode initialization."""
        corr = ZeroDspCorrelator(length=64, mode='software')
        assert not corr.using_hardware

    def test_auto_mode_fallback(self):
        """Test auto mode falls back to software."""
        corr = ZeroDspCorrelator(length=64, mode='auto')
        # Should fall back to software when hardware unavailable
        assert not corr.using_hardware

    def test_hardware_mode_fails_gracefully(self):
        """Test hardware mode fails gracefully without hardware."""
        # Hardware mode should raise if hardware unavailable
        with pytest.raises(RuntimeError):
            ZeroDspCorrelator(length=64, mode='hardware', device='/dev/nonexistent')


class TestBenchmark:
    """Test suite for benchmark utilities."""

    def test_benchmark_output_structure(self):
        """Test benchmark returns expected structure."""
        coefs = [32, 64, 96, 128]
        result = benchmark_csd_vs_multiply(coefs)
        
        assert 'coefficients' in result
        assert 'multiply_ops' in result
        assert 'csd_ops' in result
        assert 'savings_percent' in result
        assert 'details' in result

    def test_benchmark_multiply_ops(self):
        """Test multiply operations count."""
        coefs = [1, 2, 3, 4, 5]
        result = benchmark_csd_vs_multiply(coefs)
        assert result['multiply_ops'] == 5  # One multiply per coefficient


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.slow
    def test_full_radar_workflow(self):
        """Test complete radar signal processing workflow."""
        # 1. Create correlator with Barker-13
        corr = ZeroDspCorrelator(length=64, mode='software')
        corr.set_coefficients(RadarCodes.BARKER_13)
        
        # 2. Generate noisy signal with multiple pulses
        np.random.seed(123)
        signal_length = 1000
        signal = np.random.randint(-100, 100, signal_length).astype(np.int16)
        
        # Insert pulses at known locations
        pulse_locations = [100, 300, 500, 700]
        amplitude = 500
        for loc in pulse_locations:
            signal[loc:loc+13] += (RadarCodes.BARKER_13 * amplitude).astype(np.int16)
        
        # 3. Process
        output = corr.process(signal)
        
        # 4. Find peaks (simple threshold detection)
        threshold = 5000
        peaks = np.where(output > threshold)[0]
        
        # Should detect pulses near insertion points
        assert len(peaks) >= 4, f"Expected 4+ peaks, got {len(peaks)}"

    def test_csd_verilog_generation(self):
        """Test Verilog code generation for CSD multiplier."""
        verilog = CSDEncoder.generate_verilog(127, data_width=16, out_width=24)
        
        # Should contain key elements
        assert "use_dsp" in verilog.lower()
        assert "wire" in verilog or "assign" in verilog
        assert "127" in verilog  # Coefficient value in comment


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
