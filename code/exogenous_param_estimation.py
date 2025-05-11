import numpy as np
import pandas as pd
import yfinance as yf
import os

class ExogenousParamEstimation:
    @staticmethod
    def estimate_implied_r_from_parity(
        clean_calls_df: pd.DataFrame,
        clean_puts_df: pd.DataFrame,
        S0: float,
        q: float,
        T: float
    ) -> pd.DataFrame:
        """
        Estimate implied interest rates using put-call parity for matching strikes.

        Args:
            clean_calls_df (pd.DataFrame): Call data with 'strike' and 'lastPrice'.
            clean_puts_df (pd.DataFrame): Put data with 'strike' and 'lastPrice'.
            S0 (float): Spot price of the underlying.
            q (float): Dividend yield.
            T (float): Time to maturity (in years).

        Returns:
            pd.DataFrame: DataFrame with 'strike', 'call', 'put', and 'implied_r'.
        """
        merged = clean_calls_df.merge(
            clean_puts_df,
            on="strike",
            suffixes=("_call", "_put")
        )

        K = merged["strike"]
        C = merged["lastPrice_call"]
        P = merged["lastPrice_put"]

        lhs = C - P - S0 * np.exp(-q * T)
        with np.errstate(divide='ignore', invalid='ignore'):
            implied_r = -np.log(-lhs / K) / T

        result = merged[["strike"]].copy()
        result["call"] = C
        result["put"] = P
        result["implied_r"] = implied_r
        result = result.replace([np.inf, -np.inf], np.nan).dropna()

        return result
    
    @staticmethod
    def estimate_historical_dividend_yield(ticker: str, S0: float, ref_date: str) -> float:
        """
        Estimate forward dividend yield using the last 4 regular dividend payments.

        Args:
            ticker (str): Stock ticker symbol (e.g., "COST").
            S0 (float): Spot price of the stock.
            ref_date (str): Reference date in 'YYYY-MM-DD' format.

        Returns:
            float: Estimated dividend yield (e.g., 0.012 = 1.2%)
        """
        ref_str = pd.to_datetime(ref_date).strftime("%Y%m%d")
        filename = f"../data/dividends_{ticker.lower()}_{ref_str}.csv"

        if os.path.exists(filename):
            print(f"[INFO] Loading dividend data from cache: {filename}")
            dividends = pd.read_csv(filename, index_col=0).squeeze("columns")
        else:
            print(f"[INFO] Fetching dividend data from Yahoo Finance for {ticker}")
            ticker_obj = yf.Ticker(ticker)
            dividends = ticker_obj.dividends

            if dividends.empty:
                raise ValueError("No dividend data found.")

            dividends.to_csv(filename)
            print(f"[INFO] Saved dividend data to: {filename}")

        # Ensure it's a Series and clean it
        if isinstance(dividends, pd.DataFrame):
            dividends = dividends.iloc[:, 0]

        dividends = dividends.sort_index(ascending=False)
        dividends = dividends[dividends > 0]  # Filter zero or erroneous dividends

        # Try to exclude special dividends by threshold (e.g., > $5 for COST)
        regular_divs = dividends[dividends < 5.0]  # Adjust threshold as needed

        # Use last 4 regular payments to estimate forward yield
        N = min(4, len(regular_divs))
        if N == 0:
            raise ValueError("No regular dividend payments found.")

        recent_divs = regular_divs.iloc[:N]
        avg_div = recent_divs.mean()
        annualized_div = avg_div * 4  # Assuming quarterly dividends

        print(f"[INFO] Used {N} regular dividend(s), annualized estimate: {annualized_div:.4f} USD")
        q_est = annualized_div / S0
        return q_est