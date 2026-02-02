-- =============================================================================
-- QEDMMA-Lite v3.0 - Zero-DSP Correlator Testbench
-- =============================================================================
-- Copyright (C) 2026 Dr. Mladen Mešter / Nexellum
-- License: AGPL-3.0-or-later
--
-- Testbench verifies:
-- 1. Correct delay detection
-- 2. Zero DSP usage (verified in synthesis)
-- 3. Correlation peak at correct lag
-- =============================================================================

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.MATH_REAL.ALL;

entity zero_dsp_correlator_tb is
end entity;

architecture sim of zero_dsp_correlator_tb is

    -- Configuration
    constant N_SAMPLES  : integer := 64;
    constant CORR_WIDTH : integer := 8;
    constant CLK_PERIOD : time := 10 ns;
    constant TRUE_DELAY : integer := 23;  -- Expected correlation peak
    
    -- Signals
    signal clk        : std_logic := '0';
    signal rst        : std_logic := '1';
    signal x_ref      : std_logic_vector(N_SAMPLES-1 downto 0);
    signal y_in       : std_logic := '0';
    signal y_valid    : std_logic := '0';
    signal corr_out   : signed(CORR_WIDTH-1 downto 0);
    signal corr_valid : std_logic;
    
    -- Test data
    type bit_array_t is array (0 to N_SAMPLES+TRUE_DELAY+10-1) of std_logic;
    signal x_data : std_logic_vector(N_SAMPLES-1 downto 0);
    signal y_data : bit_array_t;
    
    -- Results tracking
    signal peak_value : signed(CORR_WIDTH-1 downto 0) := (others => '0');
    signal peak_index : integer := 0;
    signal sample_count : integer := 0;

begin

    -- =========================================================================
    -- DUT Instantiation
    -- =========================================================================
    DUT: entity work.zero_dsp_correlator
        generic map (
            N_SAMPLES  => N_SAMPLES,
            CORR_WIDTH => CORR_WIDTH
        )
        port map (
            clk        => clk,
            rst        => rst,
            x_ref      => x_ref,
            y_in       => y_in,
            y_valid    => y_valid,
            corr_out   => corr_out,
            corr_valid => corr_valid
        );

    -- =========================================================================
    -- Clock Generation
    -- =========================================================================
    clk <= not clk after CLK_PERIOD/2;

    -- =========================================================================
    -- Stimulus Process
    -- =========================================================================
    stim_proc: process
        variable seed1, seed2 : positive := 42;
        variable rand : real;
        
        -- LFSR-based PRN generator
        impure function prn_bit return std_logic is
            variable lfsr : std_logic_vector(15 downto 0) := x"ACE1";
            variable bit_out : std_logic;
        begin
            bit_out := lfsr(0);
            lfsr := '0' & lfsr(15 downto 1);
            lfsr(15) := lfsr(13) xor lfsr(12) xor lfsr(10) xor bit_out;
            return bit_out;
        end function;
        
    begin
        -- Initialize
        rst <= '1';
        y_valid <= '0';
        
        -- Generate reference PRN sequence
        for i in 0 to N_SAMPLES-1 loop
            uniform(seed1, seed2, rand);
            if rand > 0.5 then
                x_data(i) <= '1';
            else
                x_data(i) <= '0';
            end if;
        end loop;
        
        -- Generate y_data = delayed x_data + noise
        for i in 0 to N_SAMPLES+TRUE_DELAY+10-1 loop
            if i >= TRUE_DELAY and i < TRUE_DELAY + N_SAMPLES then
                -- Copy x_data with delay
                y_data(i) <= x_data(i - TRUE_DELAY);
            else
                -- Random before/after
                uniform(seed1, seed2, rand);
                if rand > 0.5 then
                    y_data(i) <= '1';
                else
                    y_data(i) <= '0';
                end if;
            end if;
        end loop;
        
        -- Apply reference
        x_ref <= x_data;
        
        wait for CLK_PERIOD * 5;
        rst <= '0';
        wait for CLK_PERIOD * 2;
        
        -- Stream y_data
        report "Starting correlation test...";
        report "Expected peak at index: " & integer'image(TRUE_DELAY);
        
        for i in 0 to N_SAMPLES+TRUE_DELAY+10-1 loop
            y_in <= y_data(i);
            y_valid <= '1';
            wait for CLK_PERIOD;
            
            -- Track peak
            if corr_valid = '1' then
                if corr_out > peak_value then
                    peak_value <= corr_out;
                    peak_index <= sample_count;
                end if;
                sample_count <= sample_count + 1;
            end if;
        end loop;
        
        y_valid <= '0';
        wait for CLK_PERIOD * 10;
        
        -- Report results
        report "============================================";
        report "TESTBENCH RESULTS";
        report "============================================";
        report "Peak correlation value: " & integer'image(to_integer(peak_value));
        report "Peak found at index: " & integer'image(peak_index);
        report "Expected index: " & integer'image(TRUE_DELAY);
        
        if peak_index = TRUE_DELAY then
            report "✅ TEST PASSED: Correct delay detected!" severity NOTE;
        else
            report "❌ TEST FAILED: Delay mismatch!" severity ERROR;
        end if;
        
        report "============================================";
        
        wait for CLK_PERIOD * 100;
        std.env.stop;
    end process;

end architecture;
