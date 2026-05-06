from __future__ import annotations

import pandas as pd


def event_window_alpha(attention_df: pd.DataFrame, events_df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    att = attention_df.copy()
    att["date"] = pd.to_datetime(att["date"])
    events = events_df.copy()
    events["date"] = pd.to_datetime(events["date"])
    rows = []
    for _, event in events.iterrows():
        st = event["date"] - pd.Timedelta(days=window)
        ed = event["date"] + pd.Timedelta(days=window)
        cut = att[(att["date"] >= st) & (att["date"] <= ed)].copy()
        if cut.empty:
            continue
        cut["event"] = event["event"]
        cut["event_date"] = event["date"]
        cut["days_from_event"] = (cut["date"] - event["date"]).dt.days
        rows.append(cut)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pre_post_shift(event_window_df: pd.DataFrame, pre: int = 5, post: int = 5) -> pd.DataFrame:
    rows = []
    for event_name, g in event_window_df.groupby("event"):
        pre_alpha = g[(g["days_from_event"] >= -pre) & (g["days_from_event"] < 0)]["alpha_t"].mean()
        post_alpha = g[(g["days_from_event"] > 0) & (g["days_from_event"] <= post)]["alpha_t"].mean()
        rows.append(
            {
                "event": event_name,
                "alpha_pre": float(pre_alpha),
                "alpha_post": float(post_alpha),
                "alpha_shift": float(post_alpha - pre_alpha),
            }
        )
    return pd.DataFrame(rows)

