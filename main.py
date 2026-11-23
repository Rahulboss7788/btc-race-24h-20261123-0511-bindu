"""DipSnap — mean-reversion cbBTC/USDC bot for make.fun's 24h BTC race.

Strategy (mean reversion, staged entries, time-based exits):
  * Watch a 240-minute (4h) rolling window of 1-minute closes.
  * Entry: the close has fallen at least 1.25% below the window's rolling
    high AND its z-score against the window mean is at or below -1.0.
    The dip-from-high test says the move is large in absolute terms; the
    z-score test says it is large relative to recent chop. Both must fire,
    so shallow drift-downs and noisy sideways bars do not trigger buys.
  * Stage 1 buys 70% of equity. If the dip deepens to 2.0% below the
    rolling high while stage 1 is held, stage 2 tops the position up to
    100% of equity at the cheaper price.
  * Exit (sell everything) on the first of:
      - reversion: z-score back at or above +1.0 (price has crossed the
        mean and run to the far side of the band);
      - take-profit: +1.2% above the position's average fill price;
      - stop-loss: -2.0% below the position's average fill price;
      - time: 360 bars (6h) held without any of the above.
  * After any exit, a 5-bar cooldown blocks immediate re-entry, so the
    bot cannot buy straight back into the same dip it just left.
  * Long only, at most one open position, at most one order per reply.

Costs: a full-size round trip pays ~46 bps (4 bps fee + 2 bps modelled
slippage + ~17 bps constant-product impact each way at this pool's
depth), so entries only fire on dips of 1.25% or more — several times
the round-trip cost — and exits wait for real reversion.

Determinism: decisions are pure functions of the bars seen so far.
No clock, environment, network or randomness is read anywhere.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

SYMBOL = "CBBTC"

# --- strategy parameters (tuned on the published dev window; every
# neighbouring value in the grid around them also ends the dev window
# in profit, so the edge does not hang on one tuned number) ---
WINDOW = 240            # bars in the rolling mean / std / high window
DIP_ENTRY = 0.0125      # buy when close is >= 1.25% below the rolling high
DIP_TOPUP = 0.02        # stage 2 when the dip reaches 2.0%
Z_ENTRY = -1.0          # ... and the close's z-score is at or below -1.0
Z_EXIT = 1.0            # sell when the z-score reverts to +1.0
STAGE1_BPS = 7000       # stage 1 spends 70% of equity
STAGE2_BPS = 10000      # stage 2 tops up to 100% of equity
TAKE_PROFIT = 0.012     # sell at +1.2% over the average fill price
STOP_LOSS = 0.02        # sell at -2.0% under the average fill price
MAX_HOLD = 360          # sell after 360 bars held, whatever the price
COOLDOWN = 5            # bars to wait after an exit before re-entering
MAX_SLIPPAGE_BPS = 200  # 2%: far above the ~19 bps a fill can move us
MIN_NOTIONAL = 1_000_000  # 1 USDC, in quote units


class DipSnap:
    """Keeps the rolling window and the open position's bookkeeping."""

    def __init__(self):
        self.closes = []          # close prices (int, price_scale 5), oldest first
        self.avg_entry = 0.0      # average fill price of the open position
        self.entry_qty = 0        # quantity held that the average covers
        self.entry_seq = -1       # seq of the bar the first fill landed on
        self.stage = 0            # 0 flat, 1 stage 1 held, 2 topped up
        self.cool_until = -1      # no new entry before this seq

    def _fold_fills(self, state, batch):
        """Update entry bookkeeping from the fills this batch reports."""
        for fill in batch.get("fills", []):
            if fill.get("symbol") != SYMBOL or fill.get("status") != "filled":
                continue
            if fill.get("side") == "buy":
                price = int(fill["price"])
                qty = int(fill["qty"])
                if qty <= 0:
                    continue
                if self.entry_qty <= 0:
                    self.avg_entry = float(price)
                    self.entry_qty = qty
                    self.entry_seq = state.seq  # seq of the batch carrying this fill (fill_bar is a ms timestamp, not a seq)
                    self.stage = 1
                else:
                    total = self.entry_qty + qty
                    self.avg_entry = (
                        self.avg_entry * self.entry_qty + price * qty
                    ) / total
                    self.entry_qty = total
            elif fill.get("side") == "sell":
                qty = int(fill["qty"])
                self.entry_qty = max(0, self.entry_qty - qty)
                if self.entry_qty == 0:
                    self.avg_entry = 0.0
                    self.entry_seq = -1
                    self.stage = 0
                    self.cool_until = state.seq + COOLDOWN
        # Reconcile with the account the harness reports: if it holds
        # nothing, no position bookkeeping may survive.
        held = state.positions.get(SYMBOL)
        if held is None or held.qty <= 0:
            if self.entry_qty != 0 or self.stage != 0:
                self.avg_entry = 0.0
                self.entry_qty = 0
                self.entry_seq = -1
                self.stage = 0

    def _window_stats(self):
        """Mean, standard deviation and high of the rolling window."""
        window = self.closes[-WINDOW:]
        mean = math.fsum(window) / len(window)
        var = math.fsum((c - mean) ** 2 for c in window) / len(window)
        return mean, math.sqrt(var), max(window)

    def on_batch(self, state, batch):
        self._fold_fills(state, batch)
        close = state.close(SYMBOL)
        if close is None or state.warming_up:
            return []
        self.closes.append(close)
        if len(self.closes) > WINDOW:
            # Only the rolling window is ever read; keep memory bounded.
            del self.closes[: len(self.closes) - WINDOW]
        if len(self.closes) < WINDOW:
            return []
        mean, sd, high = self._window_stats()
        if sd == 0.0 or high <= 0:
            return []
        z = (close - mean) / sd
        dip = 1.0 - close / high

        held = state.positions.get(SYMBOL)
        qty = held.qty if held is not None else 0
        if qty <= 0:
            # Flat: look for a confirmed dip.
            if state.seq < self.cool_until:
                return []
            if dip < DIP_ENTRY or z > Z_ENTRY:
                return []
            spend = min(
                state.equity * STAGE1_BPS // 10000,
                state.cash,
                state.weight_room(SYMBOL),
            )
            if spend < MIN_NOTIONAL:
                return []
            return [{
                "side": "buy",
                "symbol": SYMBOL,
                "notional_quote_units": spend,
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }]

        # Holding: exits first, then the stage-2 top-up.
        held_bars = state.seq - self.entry_seq if self.entry_seq >= 0 else 0
        if (
            z >= Z_EXIT
            or (self.avg_entry > 0.0 and close >= self.avg_entry * (1.0 + TAKE_PROFIT))
            or (self.avg_entry > 0.0 and close <= self.avg_entry * (1.0 - STOP_LOSS))
            or held_bars >= MAX_HOLD
        ):
            return [{
                "side": "sell",
                "symbol": SYMBOL,
                "qty": qty,
                "max_slippage_bps": MAX_SLIPPAGE_BPS,
            }]
        if self.stage == 1 and dip >= DIP_TOPUP:
            target = state.equity * STAGE2_BPS // 10000
            room = min(
                target - state.value(SYMBOL, qty),
                state.cash,
                state.weight_room(SYMBOL),
            )
            if room >= MIN_NOTIONAL:
                return [{
                    "side": "buy",
                    "symbol": SYMBOL,
                    "notional_quote_units": room,
                    "max_slippage_bps": MAX_SLIPPAGE_BPS,
                }]
        return []


def main():
    bot = DipSnap()
    makefun_bars.run(bot.on_batch)


if __name__ == "__main__":
    main()
