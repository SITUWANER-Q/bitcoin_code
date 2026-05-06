from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import ChaikinMoneyFlowIndicator, MFIIndicator, OnBalanceVolumeIndicator


def _garman_klass(df: pd.DataFrame) -> pd.Series:
    log_hl = np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(df["close"] / df["open"]).replace([np.inf, -np.inf], np.nan)
    return 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2


def build_numerical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df = df.sort_values("date").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["hl_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["oc_range"] = (df["open"] - df["close"]) / df["close"].replace(0, np.nan)
    df["dollar_vol"] = df["close"] * df["volume"]
    df["vwap_proxy"] = (df["high"] + df["low"] + df["close"]) / 3.0

    df["ma7"] = SMAIndicator(df["close"], window=7).sma_indicator()
    df["ma30"] = SMAIndicator(df["close"], window=30).sma_indicator()
    df["ma200"] = SMAIndicator(df["close"], window=200).sma_indicator()
    df["ema12"] = EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema26"] = EMAIndicator(df["close"], window=26).ema_indicator()

    macd = MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    df["adx"] = ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["rsi14"] = RSIIndicator(df["close"], window=14).rsi()
    stoch = StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["cci20"] = CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    df["roc10"] = ROCIndicator(df["close"], window=10).roc()

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["atr14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    df["obv"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["mfi14"] = MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=14).money_flow_index()
    df["cmf20"] = ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"], window=20).chaikin_money_flow()

    gk = _garman_klass(df).clip(lower=0)
    df["vol_gk_5d"] = gk.rolling(5).mean()
    df["vol_gk_20d"] = gk.rolling(20).mean()

    return df

