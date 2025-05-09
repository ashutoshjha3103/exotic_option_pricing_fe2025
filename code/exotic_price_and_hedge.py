import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.optimize import brentq


class Valuation:
    def __init__(self, S0, r, T, v0, kappa, theta, sigma, rho, lamb, mu_j, sigma_j):
        self.S0 = S0
        self.r = r
        self.T = T
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.lamb = lamb
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    def bs_characteristic_function(self, u, sigma):
        """
        Compute the characteristic function under the Black-Scholes model.

        Args:
            u (np.ndarray or complex): Fourier variable.
            sigma (float): Constant volatility.

        Returns:
            np.ndarray or complex: Characteristic function value.
        """
        i = 1j
        return np.exp(
            i * u * np.log(self.S0)
            + (i * u * (self.r - 0.5 * sigma**2) - 0.5 * u**2 * sigma**2) * self.T
        )

    def bates_characteristic_function(self, u):
        """
        Compute the characteristic function under the Bates model.

        Args:
            u (np.ndarray or complex): Fourier variable.

        Returns:
            np.ndarray or complex: Characteristic function value.
        """
        i = 1j
        x = np.log(self.S0)
        d = np.sqrt(
            (self.rho * self.sigma * i * u - self.kappa) ** 2
            + self.sigma**2 * (i * u + u**2)
        )
        g = (self.kappa - self.rho * self.sigma * i * u - d) / (
            self.kappa - self.rho * self.sigma * i * u + d
        )
        C = (
            self.kappa
            * self.theta
            / self.sigma**2
            * (
                (self.kappa - self.rho * self.sigma * i * u - d) * self.T
                - 2 * np.log((1 - g * np.exp(-d * self.T)) / (1 - g))
            )
        )
        D = (
            (self.kappa - self.rho * self.sigma * i * u - d)
            / self.sigma**2
            * ((1 - np.exp(-d * self.T)) / (1 - g * np.exp(-d * self.T)))
        )
        jump_term = self.lamb * self.T * (
            np.exp(i * u * self.mu_j - 0.5 * self.sigma_j**2 * u**2) - 1
        )
        return np.exp(i * u * x + C + D * self.v0 + jump_term)

    def carr_madan_price(self, alpha=1.5, N=4096, eta=0.25):
        """
        Compute European option prices using the Carr-Madan FFT method.

        Args:
            alpha (float): Dampening factor.
            N (int): Number of FFT points.
            eta (float): Grid spacing in Fourier domain.

        Returns:
            tuple: Arrays of strikes and call prices.
        """
        i = 1j
        lamb = 2 * np.pi / (N * eta)
        b = N * lamb / 2
        u = np.arange(N) * eta
        ku = -b + lamb * np.arange(N)
        phi = self.bates_characteristic_function(u - (alpha + 1) * i)
        psi = (
            np.exp(-self.r * self.T)
            * phi
            / (alpha**2 + alpha - u**2 + i * (2 * alpha + 1) * u)
        )
        weights = np.ones(N)
        weights[1:N - 1:2] = 4
        weights[2:N - 1:2] = 2
        weights *= eta / 3
        fft_input = np.exp(i * b * u) * psi * weights
        fft_output = fft(fft_input).real
        call_prices = np.exp(-alpha * ku) / np.pi * fft_output
        strikes = np.exp(ku)
        return strikes, call_prices

    def interpolate_call_price(self, K_target):
        """
        Interpolate call price for a given strike using FFT output.

        Args:
            K_target (float): Target strike price.

        Returns:
            float: Interpolated call price.
        """
        strikes, prices = self.carr_madan_price()
        interpolator = interp1d(strikes, prices, kind="cubic", bounds_error=True)
        return interpolator(K_target)


class Calibration(Valuation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calibrate_to_market(self, clean_options_df, initial_guess, bounds, q=0.008, weight_type="relative"):
        """
        Jointly calibrate Bates model parameters to calls and puts across multiple maturities.

        Args:
            clean_options_df (pd.DataFrame): Must contain 'strike', 'mid', 'option_type', 'maturity_T', 'r'.
            initial_guess (list): Initial Bates parameters.
            bounds (list): Bounds for optimization.
            q (float): Dividend yield.
            weight_type (str): 'uniform' or 'relative'.

        Returns:
            dict: Optimized parameters, final loss, success flag, and message.
        """
        strikes = clean_options_df["strike"].values
        market_prices = clean_options_df["mid"].values
        option_types = clean_options_df["option_type"].values
        maturities = clean_options_df["maturity_T"].values
        rates = clean_options_df["r"].values

        def objective(params):
            (
                self.v0,
                self.kappa,
                self.theta,
                self.sigma,
                self.rho,
                self.lamb,
                self.mu_j,
                self.sigma_j,
            ) = params

            model_prices = []

            for K, opt_type, T_i, r_i in zip(strikes, option_types, maturities, rates):
                try:
                    # Temporarily override self.T and self.r for interpolation
                    self.T = T_i
                    self.r = r_i

                    if opt_type == "call":
                        price = self.interpolate_call_price(K)
                    elif opt_type == "put":
                        call_price = self.interpolate_call_price(K)
                        price = call_price - self.S0 * np.exp(-q * T_i) + K * np.exp(-r_i * T_i)
                    else:
                        price = np.nan
                except Exception:
                    price = np.nan
                model_prices.append(price)

            model_prices = np.array(model_prices)
            mask = ~np.isnan(model_prices)
            error = model_prices[mask] - market_prices[mask]

            if weight_type == "relative":
                weights = 1 / (market_prices[mask] + 1e-6)
            else:
                weights = np.ones_like(error)

            return np.sum(weights * error**2)

        result = minimize(
            objective,
            x0=initial_guess,
            bounds=bounds,
            method="L-BFGS-B",
        )

        if result.success:
            (
                self.v0,
                self.kappa,
                self.theta,
                self.sigma,
                self.rho,
                self.lamb,
                self.mu_j,
                self.sigma_j,
            ) = result.x

        print("\nOptimized Parameters:")
        for name, val in zip(
            ["v0", "kappa", "theta", "sigma", "rho", "lambda", "mu_j", "sigma_j"],
            result.x,
        ):
            print(f"{name}: {val:.6f}")

        return {
            "optimized_params": result.x,
            "loss": result.fun,
            "success": result.success,
            "message": result.message,
        }


    def plot_calibrated_vs_market(self, clean_options_df, full_grid=False, num_strikes=100):
        """
        Plot model-calibrated call prices vs market prices (multi-maturity version).

        Args:
            clean_options_df (pd.DataFrame): Must include 'strike', 'mid', 'option_type', 'maturity_T', 'r'.
            full_grid (bool): Whether to use dense strike grid.
            num_strikes (int): Grid resolution.
        """
        call_df = clean_options_df[clean_options_df["option_type"] == "call"].copy()
        market_strikes = call_df["strike"].values
        market_prices = call_df["mid"].values
        maturities = call_df["maturity_T"].values
        rates = call_df["r"].values

        plt.figure(figsize=(10, 6))
        plt.plot(market_strikes, market_prices, 'o', label="Market Calls", color="steelblue", alpha=0.7)

        if full_grid:
            strike_grid = np.linspace(0.5 * self.S0, 1.5 * self.S0, num_strikes)
            # Use average T and r to build a smooth curve (approximate)
            avg_T = call_df["maturity_T"].mean()
            avg_r = call_df["r"].mean()
            self.T = avg_T
            self.r = avg_r

            model_prices = []
            for K in strike_grid:
                try:
                    model_prices.append(self.interpolate_call_price(K))
                except Exception:
                    model_prices.append(np.nan)

            plt.plot(strike_grid, model_prices, 'r-', label="Bates (Calibrated)", lw=2)
        else:
            # Point-by-point interpolation using row-specific T and r
            model_prices = []
            for K, T_i, r_i in zip(market_strikes, maturities, rates):
                try:
                    self.T = T_i
                    self.r = r_i
                    model_prices.append(self.interpolate_call_price(K))
                except Exception:
                    model_prices.append(np.nan)

            plt.plot(market_strikes, model_prices, 'r*', label="Bates (Calibrated)", markersize=6)

        plt.xlabel("Strike Price (K)")
        plt.ylabel("Call Price")
        plt.title("Bates Model Calibration: Market vs Calibrated Call Prices")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("../assets", exist_ok=True)
        filename = f"../assets/calibrated_vs_market_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        print(f"[Saved plot] {filename}")


    def plot_calls_and_puts_vs_model(self, clean_options_df, q=0.008):
        """
        Plot market call/put prices vs model prices after joint calibration
        (supports multiple maturities).

        Args:
            clean_options_df (pd.DataFrame): Must include 'strike', 'mid', 'option_type', 'maturity_T', 'r'.
            q (float): Dividend yield used in parity.
        """
        call_df = clean_options_df[clean_options_df["option_type"] == "call"].copy()
        put_df = clean_options_df[clean_options_df["option_type"] == "put"].copy()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Market data
        ax.scatter(call_df["strike"], call_df["mid"], color="blue", label="Market Calls", alpha=0.6)
        ax.scatter(put_df["strike"], put_df["mid"], color="green", label="Market Puts", alpha=0.6)

        # Model prices — match T and r per row
        model_calls = []
        for K, T_i, r_i in zip(call_df["strike"], call_df["maturity_T"], call_df["r"]):
            try:
                self.T = T_i
                self.r = r_i
                model_calls.append(self.interpolate_call_price(K))
            except Exception:
                model_calls.append(np.nan)

        model_puts = []
        for K, T_i, r_i in zip(put_df["strike"], put_df["maturity_T"], put_df["r"]):
            try:
                self.T = T_i
                self.r = r_i
                call_price = self.interpolate_call_price(K)
                put_price = call_price - self.S0 * np.exp(-q * T_i) + K * np.exp(-r_i * T_i)
                model_puts.append(put_price)
            except Exception:
                model_puts.append(np.nan)

        # Plot model
        ax.plot(call_df["strike"], model_calls, 'r*', label="Model Calls")
        ax.plot(put_df["strike"], model_puts, 'm*', label="Model Puts")

        ax.set_xlabel("Strike Price (K)")
        ax.set_ylabel("Option Price")
        ax.set_title("Market vs Bates Model: Calls and Puts")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        os.makedirs("../assets", exist_ok=True)
        filename = f"../assets/calls_puts_vs_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        print(f"[Saved plot] {filename}")


    def plot_residuals(self, clean_options_df, q=0.008):
        """
        Plot pricing residuals (Model - Market) for calls and puts.

        Args:
            clean_options_df (pd.DataFrame): Must contain 'strike', 'mid', 'option_type', 'maturity_T', 'r'.
            q (float): Dividend yield used in put-call parity.
        """
        strikes = clean_options_df["strike"].values
        market_prices = clean_options_df["mid"].values
        option_types = clean_options_df["option_type"].values
        maturities = clean_options_df["maturity_T"].values
        rates = clean_options_df["r"].values

        model_prices = []

        for K, opt_type, T_i, r_i in zip(strikes, option_types, maturities, rates):
            try:
                self.T = T_i
                self.r = r_i
                if opt_type == "call":
                    model_price = self.interpolate_call_price(K)
                elif opt_type == "put":
                    call_price = self.interpolate_call_price(K)
                    model_price = call_price - self.S0 * np.exp(-q * T_i) + K * np.exp(-r_i * T_i)
                else:
                    model_price = np.nan
            except Exception:
                model_price = np.nan
            model_prices.append(model_price)

        model_prices = np.array(model_prices)
        residuals = model_prices - market_prices

        plt.figure(figsize=(10, 4))
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.scatter(strikes, residuals, color='crimson', label='Residuals')
        plt.xlabel("Strike Price (K)")
        plt.ylabel("Residual (Model - Market)")
        plt.title("Bates Model Calibration Residuals")
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()

        os.makedirs("../assets", exist_ok=True)
        filename = f"../assets/calibration_residuals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        print(f"[Saved plot] {filename}")


    def plot_implied_volatility_smile_comparison(self, clean_options_df, q=0.008):
        """
        Plot implied volatility smile per maturity: Market vs Black-Scholes vs Bates model.

        Args:
            clean_options_df (pd.DataFrame): Must include 'strike', 'mid', 'option_type', 'maturity_T', 'r'.
            q (float): Dividend yield.
        """
        def bs_price(S, K, T, r, sigma, q, option_type):
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type == "put":
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            return np.nan

        def implied_vol(price, S, K, T, r, q, option_type):
            try:
                return brentq(
                    lambda sigma: bs_price(S, K, T, r, sigma, q, option_type) - price,
                    1e-4, 3.0
                )
            except Exception:
                return np.nan

        for T in sorted(clean_options_df["maturity_T"].unique()):
            subset = clean_options_df[clean_options_df["maturity_T"] == T].copy()
            r = subset["r"].iloc[0]  # consistent within each T

            strikes = subset["strike"].values
            market_prices = subset["mid"].values
            option_types = subset["option_type"].values

            market_iv = []
            bates_iv = []

            for K, mp, opt_type in zip(strikes, market_prices, option_types):
                # Market implied vol
                iv_mkt = implied_vol(mp, self.S0, K, T, r, q, opt_type)
                market_iv.append(iv_mkt)

                # Bates model implied vol
                try:
                    self.T = T
                    self.r = r
                    if opt_type == "call":
                        model_price = self.interpolate_call_price(K)
                    else:
                        call = self.interpolate_call_price(K)
                        model_price = call - self.S0 * np.exp(-q * T) + K * np.exp(-r * T)
                    iv_bates = implied_vol(model_price, self.S0, K, T, r, q, opt_type)
                except Exception:
                    iv_bates = np.nan

                # Filter extreme/unrealistic vols
                if iv_bates is not None and (iv_bates < 0.05 or iv_bates > 1.2):
                    iv_bates = np.nan
                bates_iv.append(iv_bates)

            # Plot per maturity
            plt.figure(figsize=(10, 6))
            plt.scatter(strikes, market_iv, color='green', label="Market Implied Vol", s=25)
            plt.plot(strikes, [0.25] * len(strikes), 'k--', label="Black-Scholes (σ=25%)")
            plt.scatter(strikes, bates_iv, color='red', marker='x', label="Bates Implied Vol", s=35)

            plt.xlabel("Strike Price (K)")
            plt.ylabel("Implied Volatility")
            plt.title(f"Volatility Smile (T = {T:.2f} years)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            os.makedirs("../assets", exist_ok=True)
            filename = f"../assets/iv_smile_comparison_T{int(T*100):03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename)
            print(f"[Saved plot] {filename}")



class MonteCarloExoticPricer(Calibration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def simulate_paths(self, N_paths=100000, N_steps=252, seed=42, vol_truncation="max"):
        """
        Simulate asset paths under the Bates model using Euler discretization.

        Args:
            N_paths (int): Number of simulation paths.
            N_steps (int): Number of time steps.
            seed (int): Random seed for reproducibility.
            vol_truncation (str): 'max' or 'abs' to handle negative variance.

        Returns:
            np.ndarray: Simulated stock price paths (N_paths × N_steps+1).
        """
        np.random.seed(seed)
        dt = self.T / N_steps
        S_paths = np.zeros((N_paths, N_steps + 1))
        v_paths = np.zeros((N_paths, N_steps + 1))
        S_paths[:, 0] = self.S0
        v_paths[:, 0] = self.v0
        EJ = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1

        for t in range(1, N_steps + 1):
            v_prev = v_paths[:, t - 1]
            S_prev = S_paths[:, t - 1]
            Z1 = np.random.normal(size=N_paths)
            Z2 = np.random.normal(size=N_paths)
            dW_v = Z1 * np.sqrt(dt)
            dW_s = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * np.sqrt(dt)
            v_candidate = (
                v_prev
                + self.kappa * (self.theta - v_prev) * dt
                + self.sigma * np.sqrt(np.maximum(v_prev, 0)) * dW_v
            )
            if vol_truncation == "abs":
                v_new = np.abs(v_candidate)
            elif vol_truncation == "max":
                v_new = np.maximum(v_candidate, 0)
            else:
                raise ValueError("vol_truncation must be 'abs' or 'max'")
            v_paths[:, t] = v_new
            N_jump = np.random.poisson(self.lamb * dt, size=N_paths)
            J = np.random.lognormal(self.mu_j, self.sigma_j, size=N_paths)
            log_return = (
                (self.r - self.lamb * EJ - 0.5 * v_prev) * dt
                + np.sqrt(np.maximum(v_prev, 0)) * dW_s
                + N_jump * np.log(J)
            )
            S_paths[:, t] = S_prev * np.exp(log_return)
        return S_paths

    def price_bonus_certificate(self, B, H, N_paths=100000, N_steps=252, seed=42, vol_truncation="max"):
        """
        Price the Bonus Certificate using Monte Carlo simulation.

        Args:
            B (float): Bonus level.
            H (float): Lower barrier.
            N_paths (int): Number of paths.
            N_steps (int): Time steps per path.
            seed (int): RNG seed.
            vol_truncation (str): Volatility truncation scheme.

        Returns:
            float: Discounted expected payoff.
        """
        paths = self.simulate_paths(N_paths, N_steps, seed, vol_truncation)
        S_T = paths[:, -1]
        barrier_hit = (paths <= H).any(axis=1)
        payoff = np.where(barrier_hit, S_T, np.where(S_T >= B, S_T, B))
        return np.exp(-self.r * self.T) * payoff.mean()

    def evaluate_bonus_certificate_payoffs(self, paths, B, H):
        """
        Compute per-path payoffs for the Bonus Certificate.

        Args:
            paths (np.ndarray): Simulated paths.
            B (float): Bonus level.
            H (float): Lower barrier.

        Returns:
            np.ndarray: Payoff per path.
        """
        S_T = paths[:, -1]
        barrier_hit = (paths <= H).any(axis=1)
        return np.where(barrier_hit, S_T, np.where(S_T >= B, S_T, B))

    def plot_simulated_paths(self, paths, n_paths_plot=50):
        """
        Plot a subset of simulated stock price paths.

        Args:
            paths (np.ndarray): Simulated paths (N_paths × N_steps + 1).
            n_paths_plot (int): Number of paths to display (default: 50).
        """
        time_grid = np.linspace(0, self.T, paths.shape[1])

        plt.figure(figsize=(10, 5))
        for i in range(min(n_paths_plot, paths.shape[0])):
            plt.plot(time_grid, paths[i], lw=0.6, alpha=0.7)

        plt.title(f"Simulated Asset Price Paths (T = {self.T})")
        plt.xlabel("Time (years)")
        plt.ylabel("Stock Price")
        plt.grid(True)
        plt.tight_layout()

        os.makedirs("../assets", exist_ok=True)
        filename = f"../assets/simulated_paths_T{self.T}_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(filename)
        print(f"[Saved plot] {filename}")


class HedgingStrategy:
    def __init__(self, pricer, B, H, T, notional):
        self.pricer = pricer
        self.B = B
        self.H = H
        self.T = T
        self.notional = notional
        self.S0 = pricer.S0
        self.r = pricer.r

    def compute_delta(self, eps=1.0, N_paths=100000, N_steps=252):
        """
        Compute the Delta of the Bonus Certificate using central finite differences.

        Args:
            eps (float): Perturbation in spot price.
            N_paths (int): Simulation paths.
            N_steps (int): Discretization steps.

        Returns:
            float: Estimated Delta (∂Price/∂S0).
        """
        S_up = self.S0 + eps
        S_down = self.S0 - eps
        params = (
            self.pricer.v0, self.pricer.kappa, self.pricer.theta,
            self.pricer.sigma, self.pricer.rho, self.pricer.lamb,
            self.pricer.mu_j, self.pricer.sigma_j
        )
        pricer_up = self.pricer.__class__(S_up, self.r, self.T, *params)
        pricer_down = self.pricer.__class__(S_down, self.r, self.T, *params)
        V_up = pricer_up.price_bonus_certificate(self.B, self.H, N_paths, N_steps)
        V_down = pricer_down.price_bonus_certificate(self.B, self.H, N_paths, N_steps)
        return (V_up - V_down) / (2 * eps)

    def compute_hedge_position(self, delta):
        """
        Compute number of shares needed to hedge the position.

        Args:
            delta (float): Delta of the product.

        Returns:
            float: Number of shares to hold (positive = long).
        """
        return delta * self.notional / self.S0