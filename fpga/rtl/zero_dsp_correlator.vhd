
-- QEDMMA-Lite v3.0 - Zero-DSP Correlator (VHDL)
-- Copyright (C) 2026 Dr. Mladen Mešter / Nexellum  
-- License: AGPL-3.0-or-later
-- For commercial licensing: mladen@nexellum.com

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity zero_dsp_correlator is
    generic (
        N_SAMPLES : integer := 64;  -- Reduce for synthesis test
        CORR_WIDTH : integer := 8
    );
    port (
        clk        : in  std_logic;
        rst        : in  std_logic;
        x_ref      : in  std_logic_vector(N_SAMPLES-1 downto 0);
        y_in       : in  std_logic;
        y_valid    : in  std_logic;
        corr_out   : out signed(CORR_WIDTH-1 downto 0);
        corr_valid : out std_logic
    );
end entity;

architecture rtl of zero_dsp_correlator is
    signal y_shift_reg : std_logic_vector(N_SAMPLES-1 downto 0) := (others => '0');
    signal xor_result  : std_logic_vector(N_SAMPLES-1 downto 0);
    signal hamming_cnt : unsigned(CORR_WIDTH-1 downto 0);
    
    -- Popcount function (tree adder)
    function popcount(x : std_logic_vector) return unsigned is
        variable count : unsigned(CORR_WIDTH-1 downto 0) := (others => '0');
    begin
        for i in x'range loop
            if x(i) = '1' then
                count := count + 1;
            end if;
        end loop;
        return count;
    end function;
    
begin
    -- Shift register for y samples
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                y_shift_reg <= (others => '0');
            elsif y_valid = '1' then
                y_shift_reg <= y_shift_reg(N_SAMPLES-2 downto 0) & y_in;
            end if;
        end if;
    end process;
    
    -- XOR operation (combinational)
    xor_result <= x_ref xor y_shift_reg;
    
    -- Popcount (could be pipelined for higher freq)
    hamming_cnt <= popcount(xor_result);
    
    -- Correlation output: R = N - 2*hamming
    corr_out <= to_signed(N_SAMPLES, CORR_WIDTH) - 
                signed('0' & hamming_cnt(CORR_WIDTH-2 downto 0)) - 
                signed('0' & hamming_cnt(CORR_WIDTH-2 downto 0));
                
    corr_valid <= y_valid;
    
end architecture;
