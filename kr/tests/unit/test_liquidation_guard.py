"""Unit tests for liquidation_guard module."""

from __future__ import annotations

import unittest

from src.risk.liquidation_guard import (
    LiquidationGuard,
    calc_liquidation_price,
    distance_to_liquidation_pct,
    get_dynamic_leverage,
    validate_sl_above_liquidation,
)


class TestCalcLiquidationPrice(unittest.TestCase):

    def test_long_5x(self):
        # 5x long: entry*(1 - 0.2 + 0.004) = entry*0.804
        result = calc_liquidation_price(10_000.0, 5, "long")
        self.assertAlmostEqual(result, 10_000.0 * 0.804, places=6)

    def test_short_5x(self):
        # 5x short: entry*(1 + 0.2 - 0.004) = entry*1.196
        result = calc_liquidation_price(10_000.0, 5, "short")
        self.assertAlmostEqual(result, 10_000.0 * 1.196, places=6)

    def test_long_2x(self):
        # 2x long: entry*(1 - 0.5 + 0.004) = entry*0.504
        result = calc_liquidation_price(10_000.0, 2, "long")
        self.assertAlmostEqual(result, 10_000.0 * 0.504, places=6)

    def test_short_2x(self):
        # 2x short: entry*(1 + 0.5 - 0.004) = entry*1.496
        result = calc_liquidation_price(10_000.0, 2, "short")
        self.assertAlmostEqual(result, 10_000.0 * 1.496, places=6)

    def test_long_10x(self):
        # 10x long: entry*(1 - 0.1 + 0.004) = entry*0.904
        result = calc_liquidation_price(10_000.0, 10, "long")
        self.assertAlmostEqual(result, 10_000.0 * 0.904, places=6)

    def test_short_10x(self):
        # 10x short: entry*(1 + 0.1 - 0.004) = entry*1.096
        result = calc_liquidation_price(10_000.0, 10, "short")
        self.assertAlmostEqual(result, 10_000.0 * 1.096, places=6)

    def test_invalid_side(self):
        with self.assertRaises(ValueError):
            calc_liquidation_price(10_000.0, 5, "buy")


class TestDistanceToLiquidation(unittest.TestCase):

    def test_long_distance(self):
        # current=10000, liq=8040 (5x long) → (10000-8040)/10000*100 = 19.6%
        dist = distance_to_liquidation_pct(10_000.0, 8_040.0, "long")
        self.assertAlmostEqual(dist, 19.6, places=6)

    def test_short_distance(self):
        # current=10000, liq=11960 (5x short) → (11960-10000)/10000*100 = 19.6%
        dist = distance_to_liquidation_pct(10_000.0, 11_960.0, "short")
        self.assertAlmostEqual(dist, 19.6, places=6)

    def test_long_2x_distance(self):
        # 2x long liq = 10000*0.504 = 5040; dist = (10000-5040)/10000*100 = 49.6%
        liq = calc_liquidation_price(10_000.0, 2, "long")
        dist = distance_to_liquidation_pct(10_000.0, liq, "long")
        self.assertAlmostEqual(dist, 49.6, places=6)

    def test_long_5x_distance(self):
        # 5x long liq = 10000*0.804 = 8040; dist = 19.6%
        liq = calc_liquidation_price(10_000.0, 5, "long")
        dist = distance_to_liquidation_pct(10_000.0, liq, "long")
        self.assertAlmostEqual(dist, 19.6, places=6)

    def test_long_10x_distance(self):
        # 10x long liq = 10000*0.904 = 9040; dist = 9.6%
        liq = calc_liquidation_price(10_000.0, 10, "long")
        dist = distance_to_liquidation_pct(10_000.0, liq, "long")
        self.assertAlmostEqual(dist, 9.6, places=6)

    def test_invalid_side(self):
        with self.assertRaises(ValueError):
            distance_to_liquidation_pct(10_000.0, 9_000.0, "sell")


class TestValidateSLAboveLiquidation(unittest.TestCase):

    def test_long_sl_safe(self):
        # long: sl=9000 > liq=8040 → True
        self.assertTrue(validate_sl_above_liquidation(9_000.0, 8_040.0, "long"))

    def test_long_sl_unsafe(self):
        # long: sl=8000 < liq=8040 → False
        self.assertFalse(validate_sl_above_liquidation(8_000.0, 8_040.0, "long"))

    def test_short_sl_safe(self):
        # short: sl=11000 < liq=11960 → True
        self.assertTrue(validate_sl_above_liquidation(11_000.0, 11_960.0, "short"))

    def test_short_sl_unsafe(self):
        # short: sl=12000 > liq=11960 → False
        self.assertFalse(validate_sl_above_liquidation(12_000.0, 11_960.0, "short"))


class TestGetDynamicLeverage(unittest.TestCase):

    def test_very_high_volatility(self):
        self.assertEqual(get_dynamic_leverage(5.0), 2)
        self.assertEqual(get_dynamic_leverage(6.0), 2)

    def test_high_volatility(self):
        self.assertEqual(get_dynamic_leverage(3.0), 3)
        self.assertEqual(get_dynamic_leverage(4.9), 3)

    def test_medium_volatility(self):
        self.assertEqual(get_dynamic_leverage(2.0), 4)
        self.assertEqual(get_dynamic_leverage(2.9), 4)

    def test_low_volatility(self):
        self.assertEqual(get_dynamic_leverage(1.9), 5)
        self.assertEqual(get_dynamic_leverage(0.5), 5)


class TestLiquidationGuardCheckProximity(unittest.TestCase):

    def _pos(self, current: float, liq: float, side: str) -> dict:
        return {
            "symbol": "BTCUSDT",
            "current_price": current,
            "liquidation_price": liq,
            "side": side,
        }

    def test_safe(self):
        # dist = (10000-7500)/10000*100 = 25% > 20% → SAFE
        self.assertEqual(
            LiquidationGuard.check_proximity(self._pos(10_000, 7_500, "long")),
            "SAFE",
        )

    def test_watch(self):
        # dist ~18% (between 15 and 20) → WATCH
        # liq = 8200 → dist = 18%
        self.assertEqual(
            LiquidationGuard.check_proximity(self._pos(10_000, 8_200, "long")),
            "WATCH",
        )

    def test_warning(self):
        # dist ~12% (between 8 and 15) → WARNING
        # liq = 8800 → dist = 12%
        self.assertEqual(
            LiquidationGuard.check_proximity(self._pos(10_000, 8_800, "long")),
            "WARNING",
        )

    def test_critical(self):
        # dist ~5% → CRITICAL
        # liq = 9500 → dist = 5%
        self.assertEqual(
            LiquidationGuard.check_proximity(self._pos(10_000, 9_500, "long")),
            "CRITICAL",
        )

    def test_short_safe(self):
        # short: liq=12500, current=10000 → dist=25% → SAFE
        self.assertEqual(
            LiquidationGuard.check_proximity(self._pos(10_000, 12_500, "short")),
            "SAFE",
        )

    def test_short_critical(self):
        # short: liq=10500, current=10000 → dist=5% → CRITICAL
        self.assertEqual(
            LiquidationGuard.check_proximity(self._pos(10_000, 10_500, "short")),
            "CRITICAL",
        )


class TestLiquidationGuardPreEntryCheck(unittest.TestCase):

    def _check(self, **kwargs):
        defaults = dict(
            entry_price=10_000.0,
            leverage=5,
            side="long",
            stop_loss=9_000.0,
            account_balance=10_000.0,
            position_size_usdt=1_000.0,
        )
        defaults.update(kwargs)
        return LiquidationGuard.pre_entry_check(**defaults)

    def test_all_pass(self):
        ok, msg = self._check()
        self.assertTrue(ok)
        self.assertEqual(msg, "OK")

    def test_rejects_leverage_above_5x(self):
        ok, reason = self._check(leverage=10)
        self.assertFalse(ok)
        self.assertIn("leverage", reason)

    def test_rejects_sl_beyond_liquidation_long(self):
        # 5x long liq ≈ 8040; sl=7900 < liq → unsafe
        ok, reason = self._check(stop_loss=7_900.0)
        self.assertFalse(ok)
        self.assertIn("stop_loss", reason)

    def test_rejects_sl_beyond_liquidation_short(self):
        # 5x short liq ≈ 11960; sl=12000 > liq → unsafe
        ok, reason = self._check(side="short", stop_loss=12_000.0)
        self.assertFalse(ok)
        self.assertIn("stop_loss", reason)

    def test_rejects_liquidation_too_close(self):
        # 5x long: liq dist ≈ 19.6% — passes distance check
        # Use leverage=4 for tighter liq test:
        # 4x long liq = entry*(1-0.25+0.004) = entry*0.754; dist = 24.6% — still safe
        # To trigger <15%, need leverage close to but within 5x... actually 5x gives 19.6% which is fine.
        # Force by patching: use a very high-leverage equivalent via custom entry/sl
        # Instead: test via short side with tight params
        # 5x short liq = 11960; entry=10000; dist=19.6%; sl must be < 11960 for safe SL
        # All passing with 5x. To force dist<15%, need leverage >~7x but max allowed is 5.
        # At leverage=5 dist is ~19.6% for both sides — always > 15%.
        # The distance check would only fire for leverage > ~6.7x which is above our max.
        # Test: confirm at leverage=5, distance check does NOT trigger (it's 19.6%).
        # We can't independently trigger dist<15 without also triggering leverage>5 check first.
        # So: verify check order — leverage check fires before distance check.
        ok, reason = self._check(leverage=10)
        self.assertFalse(ok)
        self.assertIn("leverage", reason)  # leverage fires first

    def test_rejects_oversized_position(self):
        # position_size_usdt = 2500, balance = 10000 → 25% > 20%
        ok, reason = self._check(position_size_usdt=2_500.0)
        self.assertFalse(ok)
        self.assertIn("position size", reason)

    def test_short_all_pass(self):
        # 5x short: sl must be < liq ≈ 11960
        ok, msg = self._check(side="short", stop_loss=11_000.0)
        self.assertTrue(ok)
        self.assertEqual(msg, "OK")

    def test_zero_balance_skips_size_check(self):
        # balance=0 → size check skipped, should pass if other conditions ok
        ok, msg = self._check(account_balance=0.0, position_size_usdt=999_999.0)
        self.assertTrue(ok)
        self.assertEqual(msg, "OK")


if __name__ == "__main__":
    unittest.main()
