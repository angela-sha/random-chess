"""Random chess: simulate games of uniformly at random chosen legal moves.
Designed for statistical analysis only.

Usage (CLI):
    python random_chess.py -n 100000 --workers 8 --seed 42 --visits --out results.csv
"""
from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import chess
import numpy as np
import pandas as pd

PIECE_NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
               chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
OUTCOMES = ["White", "Black", "Draw"]


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class GameResult:
    game_id: int
    outcome: str          # "White" | "Black" | "Draw"
    termination: str      # e.g. "CHECKMATE", "STALEMATE", "INSUFFICIENT_MATERIAL"
    plies: int            # half-moves played
    captures: int
    checks: int
    promotions: int
    material_white: int   # material left at the end (P=1 N=B=3 R=5 Q=9)
    material_black: int


def _material(board: chess.Board, color: bool) -> int:
    return sum(len(board.pieces(pt, color)) * v for pt, v in PIECE_VALUES.items())


def play_game(game_id: int = 0, rng: random.Random | None = None,
              visits: np.ndarray | None = None) -> GameResult:
    """Play one random game.

    visits: optional int array of shape (2, 64, 6) [color, square, piece_type-1]
            that is incremented in place with each move's destination square.
            Index 0 = black, 1 = white (chess.BLACK/chess.WHITE).
    """
    rng = rng or random.Random()
    board = chess.Board()
    captures = checks = promotions = 0

    while True:
        # claim_draw=False: only automatic draws (75-move, fivefold) end the game,
        # since random players never "claim" a draw.
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            break
        move = rng.choice(list(board.legal_moves))
        if visits is not None:
            piece = board.piece_type_at(move.from_square)
            visits[int(board.turn), move.to_square, piece - 1] += 1
        captures += board.is_capture(move)
        promotions += move.promotion is not None
        board.push(move)
        checks += board.is_check()

    winner = {chess.WHITE: "White", chess.BLACK: "Black", None: "Draw"}[outcome.winner]
    return GameResult(
        game_id=game_id, outcome=winner, termination=outcome.termination.name,
        plies=board.ply(), captures=captures, checks=checks, promotions=promotions,
        material_white=_material(board, chess.WHITE),
        material_black=_material(board, chess.BLACK),
    )


def _run_chunk(args):
    start_id, n, seed, track_visits = args
    rng = random.Random(seed)
    visits = np.zeros((2, 64, 6), dtype=np.int64) if track_visits else None
    results = [asdict(play_game(start_id + k, rng, visits)) for k in range(n)]
    return results, visits


def simulate(n_games: int, workers: int | None = None, seed: int | None = None,
             track_visits: bool = False, chunk_size: int = 1000):
    """Run n_games random games in parallel.

    Returns (DataFrame of per-game results, visits array or None).
    Results are reproducible for a given (seed, chunk_size), independent of workers.
    """
    workers = workers or os.cpu_count() or 1
    seeds = np.random.SeedSequence(seed).generate_state(-(-n_games // chunk_size))
    jobs = [(start, min(chunk_size, n_games - start), int(s), track_visits)
            for start, s in zip(range(0, n_games, chunk_size), seeds)]

    rows, visits = [], np.zeros((2, 64, 6), dtype=np.int64) if track_visits else None
    if workers == 1:
        outputs = map(_run_chunk, jobs)
        for r, v in outputs:
            rows.extend(r)
            if track_visits:
                visits += v
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r, v in ex.map(_run_chunk, jobs):
                rows.extend(r)
                if track_visits:
                    visits += v
    return pd.DataFrame(rows), visits


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def outcome_summary(df: pd.DataFrame, z: float = 1.96) -> pd.DataFrame:
    """Share of each outcome over all games and over decisive games, with 95% CIs."""
    n, counts = len(df), df["outcome"].value_counts()
    wins = int(counts.get("White", 0) + counts.get("Black", 0))
    rows = []
    for o in OUTCOMES:
        k = int(counts.get(o, 0))
        lo, hi = wilson_ci(k, n, z)
        row = {"outcome": o, "games": k, "share": k / n, "ci_low": lo, "ci_high": hi}
        if o != "Draw":
            row["share_of_wins"] = k / wins if wins else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("outcome")


def white_advantage_test(df: pd.DataFrame):
    """Two-sided binomial test: among decisive games, is P(White wins) != 0.5?"""
    from scipy.stats import binomtest
    w = int((df["outcome"] == "White").sum())
    b = int((df["outcome"] == "Black").sum())
    return binomtest(w, w + b, 0.5)


def termination_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Counts/shares of each termination reason, with length stats."""
    g = df.groupby("termination")["plies"]
    out = pd.DataFrame({"games": g.size(), "share": g.size() / len(df),
                        "mean_plies": g.mean(), "median_plies": g.median()})
    return out.sort_values("games", ascending=False)


def length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats of game length by outcome (plus 'All')."""
    by = df.groupby("outcome")["plies"].describe(percentiles=[.05, .25, .5, .75, .95])
    by.loc["All"] = df["plies"].describe(percentiles=[.05, .25, .5, .75, .95])
    return by


def length_kdes(df: pd.DataFrame, grid: np.ndarray | None = None):
    """Weighted KDE curves of game length per outcome (weights = outcome share),
    so the curves sum to the overall density. Returns (grid, {name: curve})."""
    from scipy.stats import gaussian_kde
    grid = np.arange(df["plies"].max() + 1) if grid is None else grid
    n = len(df)
    curves = {}
    for o in OUTCOMES:
        x = df.loc[df["outcome"] == o, "plies"]
        if len(x) > 1 and x.nunique() > 1:
            curves[o] = gaussian_kde(x)(grid) * len(x) / n
        else:
            curves[o] = np.zeros_like(grid, dtype=float)
    curves["Wins"] = curves["White"] + curves["Black"]
    curves["All"] = curves["Wins"] + curves["Draw"]
    return grid, curves


def modal_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Most frequent game length (KDE mode) for each group — the notebook's printout."""
    grid, curves = length_kdes(df)
    counts = df["outcome"].value_counts()
    n_games = {"All": len(df), "Wins": int(counts.get("White", 0) + counts.get("Black", 0)),
               **{o: int(counts.get(o, 0)) for o in OUTCOMES}}
    return pd.DataFrame([{"group": k, "mode_plies": int(grid[c.argmax()]),
                          "density": round(float(c.max()), 5), "games": n_games[k]}
                         for k, c in curves.items()]).set_index("group")


def outcome_by_length(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Outcome shares within game-length bins (how decisiveness changes with length)."""
    cut = pd.cut(df["plies"], bins=bins)
    return pd.crosstab(cut, df["outcome"], normalize="index").reindex(columns=OUTCOMES, fill_value=0)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_length_kde(df: pd.DataFrame, wins_only: bool = False, ax=None):
    """Recreates the notebook's two KDE plots (all outcomes / wins only)."""
    import matplotlib.pyplot as plt
    grid, c = length_kdes(df)
    s = outcome_summary(df)
    ax = ax or plt.subplots(figsize=(10, 5))[1]
    if wins_only:
        ax.plot(grid, c["Wins"], lw=2, label="Wins")
        for o in ("Black", "White"):
            ax.plot(grid, c[o], lw=1, label=f"{o} share of wins ≈ {s.loc[o, 'share_of_wins']:.2%}")
        ax.set_title(f"Distribution of game length, decisive games (n = {int(s.loc[['White', 'Black'], 'games'].sum())})")
    else:
        ax.plot(grid, c["All"], lw=2, label="Total")
        for o in ("Draw", "Black", "White"):
            ax.plot(grid, c[o], lw=1, label=f"{o} ≈ {s.loc[o, 'share']:.2%}")
        ax.set_title(f"Distribution of game length by outcome (n = {len(df)})")
    ax.set_xlabel("Plies (half-moves)")
    ax.set_ylabel("Density")
    ax.legend()
    return ax


def plot_termination(df: pd.DataFrame, ax=None):
    import matplotlib.pyplot as plt
    t = termination_summary(df)
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    ax.barh(t.index[::-1], t["share"][::-1])
    ax.set_xlabel("Share of games")
    ax.set_title("How random games end")
    return ax


def plot_length_ecdf(df: pd.DataFrame, ax=None):
    """Empirical CDF of game length per outcome — easier to read tails than a KDE."""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(10, 5))[1]
    for o in OUTCOMES:
        x = np.sort(df.loc[df["outcome"] == o, "plies"].to_numpy())
        if len(x):
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=o)
    ax.set_xlabel("Plies (half-moves)")
    ax.set_ylabel("P(length ≤ x)")
    ax.set_title("Game length ECDF by outcome")
    ax.legend()
    return ax


def plot_visit_heatmaps(visits: np.ndarray, color: bool = chess.WHITE,
                        normalize: bool = False, cmap: str = "inferno", savefig: str | None = None):
    """6 heatmaps (one per piece type) of destination-square visits for one colour.
    Drawn from White's perspective: rank 1 at the bottom, file a on the left."""
    import matplotlib.pyplot as plt
    data = visits[int(color)].reshape(8, 8, 6).astype(float)  # [rank, file, piece]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for p, ax in zip(range(6), axes.flat):
        board = data[:, :, p]
        if normalize and board.sum():
            board = board / board.sum()
        im = ax.imshow(board, cmap=cmap, origin="lower")
        ax.set_title(f"{'White' if color else 'Black'} {PIECE_NAMES[p + 1]} visits")
        ax.set_xticks(range(8), list("abcdefgh"))
        ax.set_yticks(range(8), range(1, 9))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if savefig:
        fig.savefig(savefig, dpi=300, transparent=True)
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", "--games", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--visits", action="store_true", help="track per-square visits")
    ap.add_argument("--out", default="results.csv", help="CSV of per-game results")
    ap.add_argument("--plots", action="store_true", help="show plots")
    a = ap.parse_args()

    df, visits = simulate(a.games, a.workers, a.seed, a.visits)
    df.to_csv(a.out, index=False)
    if visits is not None:
        np.save(os.path.splitext(a.out)[0] + "_visits.npy", visits)

    pd.set_option("display.width", 120)
    print(outcome_summary(df).round(4), "\n")
    print(termination_summary(df).round(3), "\n")
    print(modal_lengths(df), "\n")
    print(length_stats(df).round(1), "\n")
    t = white_advantage_test(df)
    print(f"White vs Black (decisive games): p = {t.pvalue:.4g}, "
          f"P(White) 95% CI = {tuple(round(x, 4) for x in t.proportion_ci())}")

    if a.plots:
        import matplotlib.pyplot as plt
        plot_length_kde(df)
        plot_length_kde(df, wins_only=True)
        plot_termination(df)
        plot_length_ecdf(df)
        if visits is not None:
            plot_visit_heatmaps(visits, chess.WHITE)
            plot_visit_heatmaps(visits, chess.BLACK)
        plt.show()


if __name__ == "__main__":
    main()
