import numpy as np
import pandas as pd
import yfinance as yf

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
        Estimate annualized dividend yield from trailing 1-year dividend payments.

        Args:
            ticker (str): Stock ticker symbol (e.g., "COST").
            S0 (float): Spot price of the stock.
            ref_date (str): Reference date in 'YYYY-MM-DD' format.

        Returns:
            float: Estimated dividend yield (q).
        """
        ticker_obj = yf.Ticker(ticker)
        dividends = ticker_obj.dividends

        if dividends.empty:
            raise ValueError("No dividend data found.")

        ref_date = pd.to_datetime(ref_date)
        one_year_ago = ref_date - pd.Timedelta(days=365)

        trailing_dividends = dividends[(dividends.index > one_year_ago) & (dividends.index <= ref_date)]
        total_dividends = trailing_dividends.sum()

        q_est = total_dividends / S0
        return q_est