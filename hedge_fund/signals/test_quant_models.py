"""Tests for the price/filings quant models — momentum, mean reversion,
insider flow, quality-value. Synthetic data, no network."""

from __future__ import annotations

import math

import pytest

from hedge_fund.data.models import FinancialMetrics, InsiderTrade
from hedge_fund.features.technicals import MONTH, YEAR
from hedge_fund.features.test_technicals import bars, linear
from hedge_fund.models import Signal
from hedge_fund.signals import (
    ALPHA_MODEL_REGISTRY,
    QUANT_MODEL_NAMES,
    InsiderFlowModel,
    MeanReversionModel,
    MomentumModel,
    QualityValueModel,
)
from hedge_fund.signals.base import QuantModel


class FakeClient:
    def __init__(self, prices=None, trades=None, metrics=None):
        self._prices = prices or []
        self._trades = trades or []
        self._metrics = metrics or []
        self.price_calls = []
        self.trade_calls = []

    def get_prices(self, ticker, start_date, end_date, **kw):
        self.price_calls.append((start_date, end_date))
        return [p for p in self._prices if start_date <= p.time[:10] <= end_date]

    def get_insider_trades(self, ticker, end_date, start_date=None, limit=1000):
        self.trade_calls.append((start_date, end_date))
        return [
            t for t in self._trades
            if t.filing_date <= end_date and (start_date is None or t.filing_date >= start_date)
        ]

    def get_financial_metrics(self, ticker, end_date, period="ttm", limit=10):
        return [m for m in self._metrics if (m.filing_date or m.report_period) <= end_date][:limit]


AS_OF = "2025-06-30"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registered_as_quant_models():
    for name in ("momentum", "mean-reversion", "insider-flow", "quality-value"):
        assert name in ALPHA_MODEL_REGISTRY
        assert issubclass(ALPHA_MODEL_REGISTRY[name], QuantModel)
        assert ALPHA_MODEL_REGISTRY[name]().name == name
        assert name in QUANT_MODEL_NAMES


def test_abstain_helper_marks_metadata():
    sig = MomentumModel()._abstain("T", AS_OF, "why")
    assert sig.value == 0.0
    assert sig.metadata["abstained"] is True
    assert "why" in sig.reasoning


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_rising_year_is_bullish(self):
        # Doubles over the year at low vol: strongly positive, but not saturated
        # to exactly 1 (tanh).
        closes = [100 * (1.003 ** i) for i in range(YEAR + 40)]
        sig = MomentumModel().predict("T", AS_OF, FakeClient(prices=bars(closes)))
        assert isinstance(sig, Signal)
        assert sig.value > 0.9
        assert sig.metadata.get("abstained") is not True
        assert sig.components["momentum_12_1"] > 0
        assert "12-1 momentum" in sig.reasoning

    def test_falling_year_is_bearish(self):
        closes = [100 * (0.997 ** i) for i in range(YEAR + 40)]
        sig = MomentumModel().predict("T", AS_OF, FakeClient(prices=bars(closes)))
        assert sig.value < -0.9

    def test_short_history_abstains(self):
        sig = MomentumModel().predict("T", AS_OF, FakeClient(prices=bars(linear(200))))
        assert sig.value == 0.0
        assert sig.metadata["abstained"] is True

    def test_point_in_time_ignores_future_bars(self):
        # Rising series ends in the future; as of an earlier date the model
        # must only see bars up to that date (the client filters by end_date,
        # and the snapshot re-filters).
        closes = [100 * (1.003 ** i) for i in range(YEAR + 200)]
        client = FakeClient(prices=bars(closes, end="2025-12-31"))
        sig = MomentumModel().predict("T", AS_OF, client)
        assert sig.metadata["last_bar_date"] <= AS_OF

    def test_scale_param_changes_decisiveness(self):
        closes = [100 * (1.0005 ** i) for i in range(YEAR + 40)]
        client = FakeClient(prices=bars(closes))
        mild = MomentumModel(scale=0.5).predict("T", AS_OF, client).value
        bold = MomentumModel(scale=2.0).predict("T", AS_OF, client).value
        assert 0 < mild < bold


# ---------------------------------------------------------------------------
# Mean reversion
# ---------------------------------------------------------------------------

class TestMeanReversion:
    @staticmethod
    def _noisy_flat(n, last_month_move):
        # Alternating small wiggles, then a straight-line last month.
        closes = [100 + (0.5 if i % 2 else -0.5) for i in range(n - MONTH)]
        end = closes[-1]
        closes += [end * (1 + last_month_move * (k + 1) / MONTH) for k in range(MONTH)]
        return closes

    def test_sharp_drop_is_a_buy(self):
        closes = self._noisy_flat(300, -0.20)
        sig = MeanReversionModel().predict("T", AS_OF, FakeClient(prices=bars(closes)))
        assert sig.value > 0.3
        assert sig.components["z_score_1m"] < -2
        assert "buying the dip" in sig.reasoning

    def test_sharp_rally_is_a_fade(self):
        closes = self._noisy_flat(300, 0.20)
        sig = MeanReversionModel().predict("T", AS_OF, FakeClient(prices=bars(closes)))
        assert sig.value < -0.3
        assert "fading the rally" in sig.reasoning

    def test_short_history_abstains(self):
        sig = MeanReversionModel().predict("T", AS_OF, FakeClient(prices=bars(linear(70))))
        assert sig.metadata["abstained"] is True

    def test_flat_history_abstains(self):
        sig = MeanReversionModel().predict("T", AS_OF, FakeClient(prices=bars([100.0] * 300)))
        assert sig.metadata["abstained"] is True
        assert "flat" in sig.reasoning

    def test_fetches_prices_once(self):
        client = FakeClient(prices=bars(self._noisy_flat(300, -0.1)))
        MeanReversionModel().predict("T", AS_OF, client)
        assert len(client.price_calls) == 1


# ---------------------------------------------------------------------------
# Insider flow
# ---------------------------------------------------------------------------

def _trade(filing_date, kind, value, name="Alice", shares=None, code=None):
    """kind: 'P' | 'S' | other Form 4 code."""
    return InsiderTrade(
        ticker="T", name=name, filing_date=filing_date,
        transaction_type=code if code is not None else kind,
        transaction_shares=shares,
        transaction_value=value,
    )


class TestInsiderFlow:
    def test_no_trades_is_real_neutral(self):
        sig = InsiderFlowModel().predict("T", AS_OF, FakeClient())
        assert sig.value == 0.0
        assert sig.metadata.get("abstained") is not True
        assert "no open-market" in sig.reasoning

    def test_cluster_buying_is_bullish(self):
        trades = [_trade("2025-06-01", "P", 100_000, name=n) for n in ("A", "B", "C", "D")]
        sig = InsiderFlowModel().predict("T", AS_OF, FakeClient(trades=trades))
        assert sig.value > 0.8
        assert sig.components["net_purchase_ratio"] == 1.0
        assert "4 open-market buys" in sig.reasoning

    def test_single_buy_is_damped(self):
        one = InsiderFlowModel().predict("T", AS_OF, FakeClient(trades=[_trade("2025-06-01", "P", 1e5)]))
        four = InsiderFlowModel().predict(
            "T", AS_OF, FakeClient(trades=[_trade("2025-06-01", "P", 1e5, name=str(i)) for i in range(4)]),
        )
        assert 0 < one.value < four.value

    def test_sales_are_down_weighted(self):
        # Equal dollars bought and sold: with sell_weight 0.5 the net is positive.
        trades = [_trade("2025-06-01", "P", 1e5), _trade("2025-06-02", "S", 1e5, name="Bob")]
        sig = InsiderFlowModel().predict("T", AS_OF, FakeClient(trades=trades))
        assert sig.value > 0
        sig_full = InsiderFlowModel(sell_weight=1.0).predict("T", AS_OF, FakeClient(trades=trades))
        assert sig_full.value == pytest.approx(0.0)

    def test_heavy_selling_is_bearish(self):
        trades = [_trade("2025-06-01", "S", 5e6, name=str(i)) for i in range(5)]
        sig = InsiderFlowModel().predict("T", AS_OF, FakeClient(trades=trades))
        assert sig.value < -0.8

    def test_grants_and_exercises_are_ignored(self):
        trades = [_trade("2025-06-01", "A", 1e6), _trade("2025-06-02", "M", 1e6), _trade("2025-06-03", "F", 1e6)]
        sig = InsiderFlowModel().predict("T", AS_OF, FakeClient(trades=trades))
        assert sig.value == 0.0
        assert sig.components["n_buys"] == 0.0 and sig.components["n_sells"] == 0.0

    def test_provider_labels_are_understood(self):
        trades = [_trade("2025-06-01", None, 1e5, code="P-Purchase"),
                  _trade("2025-06-02", None, 1e5, code="Sale", name="Bob")]
        sig = InsiderFlowModel(sell_weight=1.0).predict("T", AS_OF, FakeClient(trades=trades))
        assert sig.components["n_buys"] == 1.0 and sig.components["n_sells"] == 1.0

    def test_missing_code_falls_back_to_share_sign(self):
        trades = [_trade("2025-06-01", None, None, code="", shares=-500)]
        sig = InsiderFlowModel().predict("T", AS_OF, FakeClient(trades=trades))
        assert sig.components["n_sells"] == 1.0
        assert sig.metadata["basis"] == "counts"
        assert sig.value < 0

    def test_window_and_point_in_time(self):
        client = FakeClient(trades=[
            _trade("2025-01-01", "P", 1e6),  # outside the 90-day window
            _trade("2025-07-15", "P", 1e6),  # filed after as-of: future
        ])
        sig = InsiderFlowModel().predict("T", AS_OF, client)
        assert sig.value == 0.0
        start, end = client.trade_calls[0]
        assert end == AS_OF and start == "2025-04-01"


# ---------------------------------------------------------------------------
# Quality-value
# ---------------------------------------------------------------------------

def _metric(report_period, **kw):
    defaults = dict(ticker="T", period="ttm", filing_date=report_period)
    defaults.update(kw)
    return FinancialMetrics(report_period=report_period, **defaults)


QUARTERS = ["2025-03-31", "2024-12-31", "2024-09-30", "2024-06-30",
            "2024-03-31", "2023-12-31", "2023-09-30", "2023-06-30"]


class TestQualityValue:
    def test_cheap_and_good_is_bullish(self):
        # P/E now 10, historically 20-30; strong ROE/ROIC, no debt.
        hist = [_metric(q, price_to_earnings_ratio=10 if i == 0 else 20 + i,
                        return_on_equity=0.30, return_on_invested_capital=0.25,
                        operating_margin=0.30, debt_to_equity=0.1)
                for i, q in enumerate(QUARTERS)]
        sig = QualityValueModel().predict("T", AS_OF, FakeClient(metrics=hist))
        assert sig.value > 0.6
        assert sig.components["value.price_to_earnings_ratio"] == 1.0
        assert sig.components["quality"] > 0.5
        assert "on P/E only" in sig.reasoning

    def test_expensive_and_levered_is_bearish(self):
        hist = [_metric(q, price_to_earnings_ratio=60 if i == 0 else 15 + i,
                        return_on_equity=-0.05, operating_margin=-0.02, debt_to_equity=4.0)
                for i, q in enumerate(QUARTERS)]
        sig = QualityValueModel().predict("T", AS_OF, FakeClient(metrics=hist))
        assert sig.value < -0.5
        assert sig.components["value.price_to_earnings_ratio"] == -1.0

    def test_negative_pe_is_not_cheap(self):
        hist = [_metric(q, price_to_earnings_ratio=-5 if i == 0 else 20,
                        return_on_equity=0.1) for i, q in enumerate(QUARTERS)]
        sig = QualityValueModel().predict("T", AS_OF, FakeClient(metrics=hist))
        assert "value.price_to_earnings_ratio" not in sig.components

    def test_fcf_yield_higher_is_cheaper(self):
        hist = [_metric(q, free_cash_flow_yield=0.10 if i == 0 else 0.03)
                for i, q in enumerate(QUARTERS)]
        sig = QualityValueModel().predict("T", AS_OF, FakeClient(metrics=hist))
        assert sig.components["value.free_cash_flow_yield"] == 1.0
        assert "quality" not in sig.components  # nothing filed for quality
        assert sig.value == pytest.approx(1.0)

    def test_too_few_periods_abstains(self):
        hist = [_metric(q, price_to_earnings_ratio=10) for q in QUARTERS[:3]]
        sig = QualityValueModel().predict("T", AS_OF, FakeClient(metrics=hist))
        assert sig.metadata["abstained"] is True

    def test_nothing_filed_abstains(self):
        hist = [_metric(q) for q in QUARTERS]
        sig = QualityValueModel().predict("T", AS_OF, FakeClient(metrics=hist))
        assert sig.metadata["abstained"] is True

    def test_point_in_time_via_filing_date(self):
        # The cheap quarter was filed after as-of; the model must not see it.
        hist = [_metric("2025-06-30", price_to_earnings_ratio=5, return_on_equity=0.3)] + [
            _metric(q, price_to_earnings_ratio=20 + i, return_on_equity=0.1)
            for i, q in enumerate(QUARTERS)
        ]
        hist[0].filing_date = "2025-08-01"
        sig = QualityValueModel().predict("T", AS_OF, FakeClient(metrics=hist))
        assert sig.metadata["report_period"] == "2025-03-31"
        assert sig.components["value.price_to_earnings_ratio"] == 1.0  # 20 is the low of 20..27

    def test_value_is_bounded(self):
        hist = [_metric(q, price_to_earnings_ratio=1 if i == 0 else 100,
                        free_cash_flow_yield=0.5 if i == 0 else 0.01,
                        return_on_equity=5.0, return_on_invested_capital=5.0,
                        operating_margin=0.9, debt_to_equity=0.0)
                for i, q in enumerate(QUARTERS)]
        sig = QualityValueModel().predict("T", AS_OF, FakeClient(metrics=hist))
        assert -1.0 <= sig.value <= 1.0
        assert not math.isnan(sig.value)


# ---------------------------------------------------------------------------
# Quality-value at the current price
# ---------------------------------------------------------------------------

def _qv_rows(latest_pe, history_pes):
    """Latest row struck at P/E x EPS 1.0 = latest_pe; prior rows as filed."""
    from hedge_fund.data.models import FinancialMetrics
    dates = ["2025-03-31", "2024-12-31", "2024-09-30", "2024-06-30",
             "2024-03-31", "2023-12-31", "2023-09-30", "2023-06-30"]
    rows = [FinancialMetrics(ticker="T", report_period=dates[0], period="ttm",
                             filing_date=dates[0], earnings_per_share=1.0,
                             price_to_earnings_ratio=latest_pe)]
    for d, pe in zip(dates[1:], history_pes):
        rows.append(FinancialMetrics(ticker="T", report_period=d, period="ttm",
                                     filing_date=d, price_to_earnings_ratio=pe))
    return rows


def test_quality_value_judges_cheapness_at_todays_price():
    from hedge_fund.data.models import Price
    hist = _qv_rows(latest_pe=10.0, history_pes=[12, 14, 16, 18, 20, 22, 24])
    # As filed, P/E 10 is the cheapest the stock has been: value +1.
    cheap = QualityValueModel().predict("T", AS_OF, FakeClient(metrics=hist))
    assert cheap.components["value"] == pytest.approx(1.0)
    assert cheap.metadata["price"] is None
    # The tape has since tripled: the same EPS at $30 is P/E 30, the richest ever.
    bar = Price(open=30, close=30, high=30, low=30, volume=1, time=f"{AS_OF}T00:00:00Z")
    rich = QualityValueModel().predict("T", AS_OF, FakeClient(prices=[bar], metrics=hist))
    assert rich.components["value"] == pytest.approx(-1.0)
    assert rich.metadata["price"] == 30.0
    assert "P/E 30.0" in rich.reasoning
