import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta


class GetStockData:
    def __init__(self, ticker: str, ref_date: str):
        """
        Initialize the GetStockData object.

        Args:
            ticker (str): Ticker symbol for the stock (e.g., "COST").
            ref_date (str): Reference date in 'YYYY-MM-DD' format.
        """
        self.ticker = ticker
        self.ref_date = pd.to_datetime(ref_date)
        self.stock = yf.Ticker(self.ticker)

    def get_historical_data(self, lookback_days: int = 365) -> pd.DataFrame:
        """
        Fetch historical OHLCV data up to the reference date.

        Args:
            lookback_days (int): Number of days to look back.

        Returns:
            pd.DataFrame: Historical stock data.
        """
        start_date = self.ref_date - timedelta(days=lookback_days)
        df = self.stock.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=self.ref_date.strftime("%Y-%m-%d"),
        )
        if df.empty:
            raise ValueError("No historical data found for the specified range.")
        expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        return df[[col for col in expected if col in df.columns]]

    def get_current_price(self) -> float:
        """
        Get the spot price as of the reference date.

        Returns:
            float: Closing price on the reference date.
        """
        df = self.stock.history(
            start=self.ref_date.strftime("%Y-%m-%d"),
            end=(self.ref_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if df.empty:
            raise ValueError("No price data for the reference date.")
        return df["Close"].iloc[0]

    def get_option_chain(self) -> dict:
        """
        Fetch the option chain closest to ~3-month maturity.

        Returns:
            dict: {'calls': DataFrame, 'puts': DataFrame, 'expiry': str}
        """
        option_dates = self.stock.options
        maturity_target = self.ref_date + timedelta(days=91)
        closest_date = min(
            option_dates, key=lambda x: abs(pd.to_datetime(x) - maturity_target)
        )
        opt_chain = self.stock.option_chain(closest_date)
        return {
            "calls": opt_chain.calls,
            "puts": opt_chain.puts,
            "expiry": closest_date,
        }
    
    def get_option_chain_for_maturity(self, target_T: float) -> dict:
        """
        Fetch the option chain closest to the desired maturity in years.

        Args:
            target_T (float): Target maturity in years (e.g., 0.25, 0.5, 1.0)

        Returns:
            dict: {'calls': DataFrame, 'puts': DataFrame, 'expiry': str}
        """
        target_days = int(target_T * 365)
        option_dates = self.stock.options
        if not option_dates:
            raise ValueError("No option expiries available for this ticker.")

        # Find the expiry closest to ref_date + target_days
        target_date = self.ref_date + timedelta(days=target_days)
        closest_expiry = min(
            option_dates, key=lambda x: abs(pd.to_datetime(x) - target_date)
        )
        opt_chain = self.stock.option_chain(closest_expiry)
        return {
            "calls": opt_chain.calls.copy(),
            "puts": opt_chain.puts.copy(),
            "expiry": closest_expiry,
        }

    def check_arbitrage_conditions(
        self,
        calls_df: pd.DataFrame,
        puts_df: pd.DataFrame,
        spot: float,
        r: float,
        q: float,
        T: float,
        epsilon: float = 0.5,
        reduced_arbitrage_check: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply liquidity and arbitrage-free filtering to raw option chain data.

        When reduced_arbitrage_check=True, only basic arbitrage conditions are used:
            - Liquidity: volume or open interest > 0
            - Monotonicity: call prices should decrease, put prices increase with strike
            - Butterfly spread convexity condition

        When reduced_arbitrage_check=False, the following are also applied:
            - Mid-price computation (bid/ask or fallback to last price)
            - Put-call parity check using relative error

        Args:
            calls_df (pd.DataFrame): Call option chain.
            puts_df (pd.DataFrame): Put option chain.
            spot (float): Spot price of the stock.
            r (float): Risk-free rate.
            q (float): Dividend yield.
            T (float): Time to maturity in years.
            epsilon (float): Tolerance for monotonicity deviations (in USD).
            reduced_arbitrage_check (bool): Whether to skip parity + mid-price logic.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Cleaned calls and puts DataFrames with 'is_clean' column.
        """
        # Step 1: Liquidity filter
        calls_df = calls_df[
            (calls_df["volume"] > 0) | (calls_df["openInterest"] > 0)
        ].copy()
        puts_df = puts_df[
            (puts_df["volume"] > 0) | (puts_df["openInterest"] > 0)
        ].copy()

        if not reduced_arbitrage_check:
            # Compute mid prices with fallback
            calls_df["mid"] = (calls_df["bid"] + calls_df["ask"]) / 2
            puts_df["mid"] = (puts_df["bid"] + puts_df["ask"]) / 2

            calls_df["mid"] = np.where(
                calls_df["mid"] == 0, calls_df["lastPrice"], calls_df["mid"]
            )
            puts_df["mid"] = np.where(
                puts_df["mid"] == 0, puts_df["lastPrice"], puts_df["mid"]
            )

            # Put-Call parity check
            merged_df = calls_df.merge(puts_df, on="strike", suffixes=("_call", "_put"))
            merged_df["lhs"] = merged_df["mid_call"] - merged_df["mid_put"]
            merged_df["rhs"] = spot * np.exp(-q * T) - merged_df["strike"] * np.exp(-r * T)
            merged_df["abs_diff"] = np.abs(merged_df["lhs"] - merged_df["rhs"])
            merged_df["relative_error"] = merged_df["abs_diff"] / (
                np.abs(merged_df["rhs"]) + 1e-6
            )
            merged_df["parity_clean"] = merged_df["relative_error"] < 0.15
            valid_strikes = set(merged_df[merged_df["parity_clean"]]["strike"])
        else:
            # Skip mid/last price fallback — use lastPrice only
            calls_df["mid"] = calls_df["lastPrice"]
            puts_df["mid"] = puts_df["lastPrice"]
            valid_strikes = set(calls_df["strike"]).intersection(set(puts_df["strike"]))

        # Step 2: Monotonicity
        calls_df = calls_df.sort_values("strike").reset_index(drop=True)
        puts_df = puts_df.sort_values("strike").reset_index(drop=True)

        calls_df["monotonic_clean"] = True
        puts_df["monotonic_clean"] = True

        calls_diff = calls_df["mid"].diff()
        puts_diff = puts_df["mid"].diff()

        calls_df.loc[1:, "monotonic_clean"] = calls_diff[1:] <= epsilon
        puts_df.loc[1:, "monotonic_clean"] = puts_diff[1:] >= -epsilon

        # Step 3: Butterfly convexity
        def butterfly_violation(df):
            df_sorted = df.sort_values("strike").reset_index(drop=True)
            clean = [True] * len(df_sorted)
            for i in range(1, len(df_sorted) - 1):
                p1 = df_sorted.loc[i - 1, "mid"]
                p2 = df_sorted.loc[i, "mid"]
                p3 = df_sorted.loc[i + 1, "mid"]
                if p2 > 0.5 * (p1 + p3):
                    clean[i] = False
            return clean

        calls_df["butterfly_clean"] = butterfly_violation(calls_df)
        puts_df["butterfly_clean"] = butterfly_violation(puts_df)

        # Step 4: Combine
        calls_df["is_clean"] = (
            calls_df["monotonic_clean"]
            & calls_df["butterfly_clean"]
            & calls_df["strike"].isin(valid_strikes)
        )
        puts_df["is_clean"] = (
            puts_df["monotonic_clean"]
            & puts_df["butterfly_clean"]
            & puts_df["strike"].isin(valid_strikes)
        )

        # Step 5: Summary
        print("=== No-Arbitrage Check Summary ===")
        print(f"Call monotonicity clean: {calls_df['monotonic_clean'].sum()} / {len(calls_df)}")
        print(f"Put monotonicity clean: {puts_df['monotonic_clean'].sum()} / {len(puts_df)}")
        print(f"Call butterfly clean: {calls_df['butterfly_clean'].sum()} / {len(calls_df)}")
        print(f"Put butterfly clean: {puts_df['butterfly_clean'].sum()} / {len(puts_df)}")
        print(f"Calls retained (all clean): {calls_df['is_clean'].sum()} / {len(calls_df)}")
        print(f"Puts retained (all clean): {puts_df['is_clean'].sum()} / {len(puts_df)}")

        return calls_df, puts_df
