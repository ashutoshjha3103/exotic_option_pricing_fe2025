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
    
    def estimate_historical_dividend_yield(ticker: str, S0: float, ref_date: str) -> float:
        """
        Estimate dividend yield from the last up to 8 dividend payments.

        Args:
            ticker (str): Stock ticker symbol (e.g., "COST").
            S0 (float): Spot price of the stock.
            ref_date (str): Reference date in 'YYYY-MM-DD' format.

        Returns:
            float: Estimated dividend yield (e.g., 0.025 = 2.5%)
        """
        ref_str = pd.to_datetime(ref_date).strftime("%Y%m%d")
        filename = f"../data/dividends_{ticker.lower()}_{ref_str}.csv"

        # Load or fetch dividend data
        if os.path.exists(filename):
            print(f"[INFO] Loading dividend data from cache: {filename}")
            dividends = pd.read_csv(filename, index_col=0).squeeze("columns")
        else:
            print(f"[INFO] Fetching dividend data from Yahoo Finance for {ticker}")
            ticker_obj = yf.Ticker(ticker)
            dividends = ticker_obj.dividends

            if dividends.empty:
                raise ValueError("No dividend data found.")

            # Save to CSV and drop datetime complications
            dividends = dividends.reset_index(drop=True)
            dividends.to_csv(filename)
            print(f"[INFO] Saved dividend data to: {filename}")

        # Ensure we have a Series
        if isinstance(dividends, pd.DataFrame):
            dividends = dividends.iloc[:, 0]

        dividends = dividends.reset_index(drop=True)

        # Use last 8 payments or all available if fewer
        N = min(8, len(dividends))
        recent_divs = dividends.tail(N)
        total_dividends = recent_divs.sum()

        print(f"[INFO] Used last {N} dividend(s), total paid: {total_dividends:.4f} USD")
        q_est = total_dividends / S0
        return q_est