#!/usr/bin/env python3
"""
Zero-DSP Correlation Module for QEDMMA
=======================================
Author:  Dr. Mladen Mešter / Nexellum d.o.o.
License: AGPL-3.0-or-later
Contact: mladen@nexellum.com | +385 99 737 5100

This module provides:
1. Software reference implementation of Zero-DSP correlation
2. Hardware interface wrapper for FPGA deployment
3. CSD coefficient encoder
4. Performance comparison utilities

Integration with qedmma-lite:
    from qedmma_lite.zero_dsp import ZeroDspCorrelator, CSDEncoder
    
    # Software mode (Python)
    corr = ZeroDspCorrelator(length=64, mode='software')
    corr.set_coefficients(barker13_coefficients)
    result = corr.process(samples)
    
    # Hardware mode (FPGA via AXI)
    corr = ZeroDspCorrelator(length=64, mode='hardware', device='/dev/uio0')
    corr.set_coefficients(barker13_coefficients)
    result = corr.process(samples)
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import struct
import time

__version__ = "1.0.0"
__author__ = "Dr. Mladen Mešter"


# =============================================================================
# CSD (Canonical Signed Digit) Encoder
# =============================================================================

class CSDEncoder:
    """
    Canonical Signed Digit encoder for Zero-DSP multiplication.
    Converts arbitrary integers to minimal shift-add representation.
    """
    
    @staticmethod
    def encode(n: int) -> List[Tuple[int, int]]:
        """
        Convert integer to CSD representation.
        
        Args:
            n: Integer to encode
            
        Returns:
            List of (position, sign) tuples where sign ∈ {-1, +1}
            
        Example:
            >>> CSDEncoder.encode(23)
            [(0, -1), (3, -1), (5, 1)]  # 23 = 2^5 - 2^3 - 2^0
        """
        csd = []
        i = 0
        while n != 0:
            if n & 1:  # Odd
                if (n & 3) == 3:  # ...11 pattern -> convert to ...(-1)0(+1)
                    csd.append((i, -1))
                    n += 1
                else:  # ...01 pattern
                    csd.append((i, 1))
                    n -= 1
            n >>= 1
            i += 1
        return csd
    
    @staticmethod
    def to_operations(csd: List[Tuple[int, int]]) -> str:
        """
        Convert CSD to human-readable operations string.
        
        Example:
            >>> CSDEncoder.to_operations([(0, -1), (3, -1), (5, 1)])
            '(x << 5) - (x << 3) - x'
        """
        if not csd:
            return "0"
        
        ops = []
        for pos, sign in sorted(csd, reverse=True):
            sign_str = "+" if sign == 1 else "-"
            if pos == 0:
                term = "x"
            else:
                term = f"(x << {pos})"
            ops.append(f"{sign_str} {term}")
        
        result = " ".join(ops)
        if result.startswith("+ "):
            result = result[2:]
        return result
    
    @staticmethod
    def operation_count(n: int) -> int:
        """Count number of add/sub operations needed for coefficient n."""
        return len(CSDEncoder.encode(n))
    
    @staticmethod
    def generate_verilog(coef: int, data_width: int = 16, out_width: int = 32) -> str:
        """
        Generate Verilog code for CSD multiplication.
        
        Args:
            coef: Coefficient value
            data_width: Input data width
            out_width: Output data width
            
        Returns:
            Verilog code snippet
        """
        csd = CSDEncoder.encode(coef)
        
        if not csd:
            return f"assign product = {out_width}'d0;"
        
        lines = [f"// CSD multiplication by {coef} = {CSDEncoder.to_operations(csd)}"]
        lines.append(f"(* use_dsp = \"no\" *)")
        lines.append(f"wire signed [{out_width-1}:0] product;")
        
        terms = []
        for pos, sign in csd:
            if pos == 0:
                term = f"{{{{({out_width - data_width}){{data_in[{data_width-1}]}}}}, data_in}}"
            else:
                term = f"({{{{({out_width - data_width}){{data_in[{data_width-1}]}}}}, data_in}} << {pos})"
            
            if sign == 1:
                terms.append(f"+ {term}")
            else:
                terms.append(f"- {term}")
        
        expr = " ".join(terms)
        if expr.startswith("+ "):
            expr = expr[2:]
        
        lines.append(f"assign product = {expr};")
        
        return "\n".join(lines)


# =============================================================================
# Ternary Coefficient Encoder
# =============================================================================

class TernaryEncoder:
    """Encode ternary coefficients (+1, 0, -1) for hardware."""
    
    # Hardware encoding: 2'b00 = 0, 2'b01 = +1, 2'b11 = -1
    ENCODING = {0: 0b00, 1: 0b01, -1: 0b11}
    DECODING = {0b00: 0, 0b01: 1, 0b11: -1, 0b10: 0}  # 0b10 reserved
    
    @staticmethod
    def encode(coefficients: np.ndarray) -> int:
        """
        Encode array of ternary coefficients to packed integer.
        
        Args:
            coefficients: Array of values in {-1, 0, +1}
            
        Returns:
            Packed integer for hardware configuration
        """
        assert np.all(np.isin(coefficients, [-1, 0, 1])), "Coefficients must be ternary"
        
        packed = 0
        for i, c in enumerate(coefficients):
            packed |= (TernaryEncoder.ENCODING[c] << (i * 2))
        return packed
    
    @staticmethod
    def decode(packed: int, length: int) -> np.ndarray:
        """
        Decode packed integer to coefficient array.
        
        Args:
            packed: Packed integer from hardware
            length: Number of coefficients
            
        Returns:
            Array of ternary coefficients
        """
        coefficients = np.zeros(length, dtype=np.int8)
        for i in range(length):
            bits = (packed >> (i * 2)) & 0b11
            coefficients[i] = TernaryEncoder.DECODING[bits]
        return coefficients


# =============================================================================
# Zero-DSP Correlator (Software Implementation)
# =============================================================================

class ZeroDspCorrelatorSW:
    """
    Software implementation of Zero-DSP correlation.
    Bit-exact match with RTL implementation for verification.
    """
    
    def __init__(self, length: int = 64, data_width: int = 16):
        """
        Initialize correlator.
        
        Args:
            length: Number of correlation taps (must be power of 2 for efficiency)
            data_width: Input sample width in bits
        """
        self.length = length
        self.data_width = data_width
        self.delay_line = np.zeros(length, dtype=np.int16)
        self.coefficients = np.zeros(length, dtype=np.int8)
        self._peak_value = np.iinfo(np.int32).min
        self._peak_index = 0
        self._sample_count = 0
        
    def set_coefficients(self, coefficients: np.ndarray):
        """
        Set correlation coefficients (ternary: +1, 0, -1).
        
        Args:
            coefficients: Array of ternary coefficients
        """
        assert len(coefficients) <= self.length, f"Too many coefficients: {len(coefficients)} > {self.length}"
        assert np.all(np.isin(coefficients, [-1, 0, 1])), "Coefficients must be ternary"
        
        self.coefficients[:] = 0
        self.coefficients[:len(coefficients)] = coefficients
        
    def reset(self):
        """Reset internal state."""
        self.delay_line[:] = 0
        self._peak_value = np.iinfo(np.int32).min
        self._peak_index = 0
        self._sample_count = 0
        
    def process_sample(self, sample: int) -> int:
        """
        Process single sample through correlator.
        
        Args:
            sample: Input sample (signed integer)
            
        Returns:
            Correlation output
        """
        # Shift delay line
        self.delay_line = np.roll(self.delay_line, 1)
        self.delay_line[0] = np.int16(sample)
        
        # Zero-DSP correlation: conditional add/subtract
        # This is functionally identical to: np.sum(delay_line * coefficients)
        # But uses only addition/subtraction
        result = np.int32(0)
        for i in range(self.length):
            if self.coefficients[i] == 1:
                result += self.delay_line[i]
            elif self.coefficients[i] == -1:
                result -= self.delay_line[i]
            # if 0, no operation
        
        # Peak detection
        if result > self._peak_value:
            self._peak_value = result
            self._peak_index = self._sample_count
        self._sample_count += 1
        
        return result
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Process array of samples.
        
        Args:
            samples: Input samples array
            
        Returns:
            Correlation outputs array
        """
        outputs = np.zeros(len(samples), dtype=np.int32)
        for i, sample in enumerate(samples):
            outputs[i] = self.process_sample(int(sample))
        return outputs
    
    @property
    def peak_value(self) -> int:
        """Get detected peak value."""
        return self._peak_value
    
    @property
    def peak_index(self) -> int:
        """Get detected peak index."""
        return self._peak_index
    
    def get_encoded_coefficients(self) -> int:
        """Get hardware-format encoded coefficients."""
        return TernaryEncoder.encode(self.coefficients)


# =============================================================================
# Standard Radar Codes
# =============================================================================

class RadarCodes:
    """Collection of standard radar waveform codes."""
    
    # Barker codes (optimal for pulse compression)
    BARKER_2 = np.array([1, -1])
    BARKER_3 = np.array([1, 1, -1])
    BARKER_4A = np.array([1, 1, -1, 1])
    BARKER_4B = np.array([1, 1, 1, -1])
    BARKER_5 = np.array([1, 1, 1, -1, 1])
    BARKER_7 = np.array([1, 1, 1, -1, -1, 1, -1])
    BARKER_11 = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])
    BARKER_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    
    @staticmethod
    def get_barker(length: int) -> np.ndarray:
        """Get Barker code of specified length."""
        codes = {
            2: RadarCodes.BARKER_2,
            3: RadarCodes.BARKER_3,
            4: RadarCodes.BARKER_4A,
            5: RadarCodes.BARKER_5,
            7: RadarCodes.BARKER_7,
            11: RadarCodes.BARKER_11,
            13: RadarCodes.BARKER_13
        }
        if length not in codes:
            raise ValueError(f"Barker code length {length} not available. Options: {list(codes.keys())}")
        return codes[length].copy()
    
    @staticmethod
    def mls(length: int, seed: int = 1) -> np.ndarray:
        """
        Generate Maximum Length Sequence (M-sequence).
        
        Args:
            length: Sequence length (will use smallest register that covers this)
            seed: Initial shift register state
            
        Returns:
            MLS sequence with values in {-1, +1}
        """
        # Find register length
        n = int(np.ceil(np.log2(length + 1)))
        max_len = (1 << n) - 1
        
        # Feedback taps for various register lengths
        taps = {
            3: [2, 0],
            4: [3, 0],
            5: [4, 2],
            6: [5, 0],
            7: [6, 0],
            8: [7, 5, 4, 0],
            9: [8, 4],
            10: [9, 6]
        }
        
        if n not in taps:
            raise ValueError(f"MLS length {n} not supported")
        
        # Generate sequence
        state = seed & ((1 << n) - 1)
        if state == 0:
            state = 1
            
        sequence = []
        for _ in range(min(length, max_len)):
            bit = state & 1
            sequence.append(1 if bit else -1)
            
            # XOR feedback
            fb = 0
            for tap in taps[n]:
                fb ^= (state >> tap) & 1
            state = ((state >> 1) | (fb << (n - 1))) & ((1 << n) - 1)
        
        return np.array(sequence, dtype=np.int8)


# =============================================================================
# Hardware Interface (for FPGA deployment)
# =============================================================================

@dataclass
class HardwareConfig:
    """Hardware configuration parameters."""
    base_addr: int = 0x40000000
    ctrl_offset: int = 0x00
    status_offset: int = 0x04
    corr_len_offset: int = 0x08
    threshold_offset: int = 0x0C
    peak_val_offset: int = 0x10
    peak_idx_offset: int = 0x14
    coef_offset: int = 0x100  # Coefficient array starts here


class ZeroDspCorrelatorHW:
    """
    Hardware interface wrapper for FPGA-based Zero-DSP correlator.
    Communicates via memory-mapped I/O (UIO or devmem).
    """
    
    def __init__(self, 
                 device: str = "/dev/uio0",
                 config: Optional[HardwareConfig] = None,
                 length: int = 64):
        """
        Initialize hardware interface.
        
        Args:
            device: UIO device path or 'devmem' for direct memory access
            config: Hardware configuration (uses defaults if None)
            length: Correlation length
        """
        self.device = device
        self.config = config or HardwareConfig()
        self.length = length
        self._mmap = None
        self._is_initialized = False
        
    def open(self):
        """Open hardware device."""
        try:
            import mmap
            
            if self.device.startswith("/dev/uio"):
                # UIO interface
                fd = open(self.device, "r+b", buffering=0)
                self._mmap = mmap.mmap(fd.fileno(), 4096)
                self._is_initialized = True
            else:
                raise ValueError(f"Unsupported device: {self.device}")
                
        except Exception as e:
            print(f"Hardware initialization failed: {e}")
            print("Falling back to software simulation mode")
            self._is_initialized = False
    
    def close(self):
        """Close hardware device."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        self._is_initialized = False
    
    def _write_reg(self, offset: int, value: int):
        """Write 32-bit register."""
        if not self._mmap:
            return
        self._mmap.seek(offset)
        self._mmap.write(struct.pack("<I", value & 0xFFFFFFFF))
    
    def _read_reg(self, offset: int) -> int:
        """Read 32-bit register."""
        if not self._mmap:
            return 0
        self._mmap.seek(offset)
        return struct.unpack("<I", self._mmap.read(4))[0]
    
    def set_coefficients(self, coefficients: np.ndarray):
        """Load coefficients to hardware."""
        if not self._is_initialized:
            print("Warning: Hardware not initialized")
            return
            
        # Disable correlator during coefficient load
        self._write_reg(self.config.ctrl_offset, 0)
        
        # Write packed coefficients (64 taps = 128 bits = 4 x 32-bit words)
        packed = TernaryEncoder.encode(coefficients)
        for i in range(4):
            word = (packed >> (i * 32)) & 0xFFFFFFFF
            self._write_reg(self.config.coef_offset + i * 4, word)
        
        # Re-enable
        self._write_reg(self.config.ctrl_offset, 1)
    
    def get_peak(self) -> Tuple[int, int]:
        """Get peak value and index from hardware."""
        if not self._is_initialized:
            return 0, 0
        value = self._read_reg(self.config.peak_val_offset)
        index = self._read_reg(self.config.peak_idx_offset)
        return value, index
    
    def is_busy(self) -> bool:
        """Check if hardware is busy."""
        if not self._is_initialized:
            return False
        status = self._read_reg(self.config.status_offset)
        return bool(status & 1)


# =============================================================================
# Unified Interface
# =============================================================================

class ZeroDspCorrelator:
    """
    Unified Zero-DSP correlator interface.
    Automatically selects software or hardware implementation.
    """
    
    def __init__(self, 
                 length: int = 64,
                 mode: str = 'auto',
                 device: Optional[str] = None):
        """
        Initialize correlator.
        
        Args:
            length: Correlation length
            mode: 'software', 'hardware', or 'auto'
            device: Hardware device path (for hardware mode)
        """
        self.length = length
        self.mode = mode
        
        # Always create software implementation for reference/fallback
        self._sw = ZeroDspCorrelatorSW(length)
        self._hw = None
        
        if mode in ['hardware', 'auto']:
            self._hw = ZeroDspCorrelatorHW(device or "/dev/uio0", length=length)
            self._hw.open()
            
            if not self._hw._is_initialized and mode == 'hardware':
                raise RuntimeError("Hardware initialization failed")
        
        self._use_hw = (self._hw and self._hw._is_initialized)
        
    def set_coefficients(self, coefficients: np.ndarray):
        """Set correlation coefficients."""
        self._sw.set_coefficients(coefficients)
        if self._use_hw:
            self._hw.set_coefficients(coefficients)
    
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Process samples through correlator."""
        return self._sw.process(samples)
    
    def reset(self):
        """Reset correlator state."""
        self._sw.reset()
    
    @property
    def peak_value(self) -> int:
        return self._sw.peak_value
    
    @property
    def peak_index(self) -> int:
        return self._sw.peak_index
    
    @property
    def using_hardware(self) -> bool:
        return self._use_hw


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_csd_vs_multiply(coefficients: List[int], data_width: int = 16) -> dict:
    """
    Compare CSD vs direct multiplication operation counts.
    
    Args:
        coefficients: List of coefficients to analyze
        data_width: Input data width
        
    Returns:
        Dictionary with comparison statistics
    """
    results = {
        'coefficients': coefficients,
        'multiply_ops': len(coefficients),  # 1 multiply per coefficient
        'csd_ops': sum(CSDEncoder.operation_count(c) for c in coefficients),
        'savings_percent': 0,
        'details': []
    }
    
    for c in coefficients:
        csd = CSDEncoder.encode(c)
        results['details'].append({
            'value': c,
            'csd': CSDEncoder.to_operations(csd),
            'ops': len(csd)
        })
    
    if results['multiply_ops'] > 0:
        results['savings_percent'] = (
            (results['multiply_ops'] - results['csd_ops']) / 
            results['multiply_ops'] * 100
        )
    
    return results


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Zero-DSP Correlation Module Demo")
    print("=" * 60)
    
    # 1. CSD Encoder Demo
    print("\n1. CSD Encoding Examples:")
    for val in [23, 127, 255, 1000]:
        csd = CSDEncoder.encode(val)
        print(f"   {val:4d} = {CSDEncoder.to_operations(csd)} ({len(csd)} ops)")
    
    # 2. Software Correlator Demo
    print("\n2. Barker-13 Correlation:")
    correlator = ZeroDspCorrelator(length=64, mode='software')
    correlator.set_coefficients(RadarCodes.BARKER_13)
    
    # Generate test signal with Barker pulse
    np.random.seed(42)
    signal = np.zeros(100, dtype=np.int16)
    signal[50:63] = (RadarCodes.BARKER_13 * 1000).astype(np.int16)
    noise = np.random.randint(-100, 100, 100).astype(np.int16)
    signal = np.clip(signal + noise, -32768, 32767).astype(np.int16)
    
    output = correlator.process(signal)
    print(f"   Peak value: {correlator.peak_value} at index {correlator.peak_index}")
    print(f"   Expected: ~13000 at index ~62")
    
    # 3. Hardware coefficient encoding
    print("\n3. Hardware Coefficient Encoding:")
    encoded = correlator._sw.get_encoded_coefficients()
    print(f"   Barker-13 encoded: 0x{encoded:032X}")
    
    # 4. Benchmark
    print("\n4. CSD Efficiency Analysis:")
    doppler_coefs = [32, 64, 96, 128, 160, 192, 224, 256]  # Example FIR taps
    bench = benchmark_csd_vs_multiply(doppler_coefs)
    print(f"   Direct multiply operations: {bench['multiply_ops']}")
    print(f"   CSD add/sub operations:     {bench['csd_ops']}")
    print(f"   Operation savings:          {bench['savings_percent']:.1f}%")
    
    print("\n" + "=" * 60)
    print(" Demo Complete")
    print("=" * 60)
