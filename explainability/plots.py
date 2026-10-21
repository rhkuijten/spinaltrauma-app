"""SHAP plots: global importance bar, beeswarm, and per-patient waterfall.

The global plots are on the log-odds scale, where the linear model's SHAP values
are exact and additive. The waterfall reports absolute risk instead, so that a
clinician reads percentage points rather than log-odds; see
_probability_contributions() for how the values are put on that scale.

VENDORED, DO NOT EDIT HERE. Copied verbatim from the Spine-Trauma training
repo (explainability/), so the figure the app shows is the same figure the
manuscript reports. Re-copy if it changes there.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

CRIMSON = "#E8175D"
_XLABEL = "Impact on predicted log-odds of 30-day mortality"
_LABEL_SIZE = 11

# Waterfall palette: raises risk / lowers risk.
_UP, _DOWN = "#2E8B2E", "#F08080"


def _normalize_fonts(ax) -> None:
    """Force one label size on an axes; shap's own plots hard-code a larger one."""
    ax.tick_params(labelsize=_LABEL_SIZE)
    ax.xaxis.label.set_size(_LABEL_SIZE)
    ax.yaxis.label.set_size(_LABEL_SIZE)


def _save(fig, save_path: Path | None) -> None:
    """Write fig to save_path (creating parents) and close it, or leave it open."""
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_importance_bar(expl, ax=None, top_n: int = 15, save_path=None):
    """Horizontal bar chart of mean |SHAP| per feature (panel A of the summary).

    Args:
        expl: Explanation from compute_shap().
        ax: Optional axes to draw into; a new figure is created when None.
        top_n: Number of highest-ranked features to show.
        save_path: Optional PNG destination (only used when ax is None).

    Returns:
        The axes drawn into.
    """
    imp = np.abs(expl.values).mean(axis=0)
    order = np.argsort(imp)[::-1][:top_n][::-1]  # ascending: barh draws bottom-up
    names = [expl.feature_names[i] for i in order]
    vals = imp[order]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.38 * len(order) + 1.2))

    ax.barh(names, vals, color=CRIMSON, height=0.62)
    for y, v in enumerate(vals):
        ax.text(v + vals.max() * 0.015, y, f"{v:.2f}",
                va="center", fontsize=9, color=CRIMSON)

    ax.set_xlim(0, vals.max() * 1.14)
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _normalize_fonts(ax)

    if fig is not None:
        _save(fig, save_path)
    return ax


def plot_beeswarm(expl, ax=None, top_n: int = 15, save_path=None):
    """Beeswarm of per-patient SHAP values coloured by feature value (panel B).

    Features are ordered by mean |SHAP|, matching plot_importance_bar().

    Args:
        expl: Explanation from compute_shap().
        ax: Optional axes to draw into; a new figure is created when None.
        top_n: Number of highest-ranked features to show.
        save_path: Optional PNG destination (only used when ax is None).

    Returns:
        The axes drawn into.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.38 * min(top_n, len(expl.feature_names)) + 1.2))

    shap.plots.beeswarm(
        expl, max_display=top_n, ax=ax, show=False, plot_size=None,
        group_remaining_features=False,
        color_bar_label="Predictor input value",
    )
    ax.set_xlabel(_XLABEL)
    _normalize_fonts(ax)

    if fig is not None:
        _save(fig, save_path)
    return ax


def plot_global_summary(expl, top_n: int = 15, save_path=None):
    """Two-panel figure: A = mean |SHAP| bar chart, B = beeswarm.

    Args:
        expl: Explanation from compute_shap().
        top_n: Number of highest-ranked features to show in both panels.
        save_path: Optional PNG destination.

    Returns:
        The created figure.
    """
    n_rows = min(top_n, len(expl.feature_names))
    fig, axes = plt.subplots(
        2, 1, figsize=(9, 2 * (0.38 * n_rows + 1.4)),
    )
    plot_importance_bar(expl, ax=axes[0], top_n=top_n)
    plot_beeswarm(expl, ax=axes[1], top_n=top_n)
    for ax, letter in zip(axes, "AB"):
        ax.set_title(letter, loc="center", fontweight="bold", fontsize=13)
    fig.tight_layout(h_pad=2.0)

    # The beeswarm reserves room for its colour bar, so panel B ends up narrower
    # than panel A; match A to B so the two feature-name columns line up.
    pos_a, pos_b = axes[0].get_position(), axes[1].get_position()
    axes[0].set_position([pos_b.x0, pos_a.y0, pos_b.width, pos_a.height])

    _save(fig, save_path)
    return fig


def _integer_split(values: np.ndarray, total: int) -> np.ndarray:
    """Round values to integers summing to total, without flipping any sign.

    Rounds half away from zero, then pushes the leftover units onto the largest
    contributions one at a time. An adjustment is skipped if it would give a row
    the opposite sign to its true value, which would otherwise print "+0%" beside
    a bar drawn in the colour for a decrease.

    Args:
        values: Contributions in percentage points, summing to total.
        total: The integer the rounded values must add up to.

    Returns:
        Integer array of the same length, summing to total where the sign
        constraint allows.
    """
    out = (np.sign(values) * np.floor(np.abs(values) + 0.5)).astype(int)
    drift = total - out.sum()
    if drift:
        step = int(np.sign(drift))
        # Spend the leftover units on the rows that were rounded furthest in the
        # opposite direction, one unit each, so no row moves by a whole point.
        residual = values - out
        for i in np.argsort(-residual if step > 0 else residual):
            if not drift:
                break
            if values[i] and np.sign(out[i] + step) == -np.sign(values[i]):
                continue
            out[i] += step
            drift -= step
    return out


def _probability_contributions(expl, index: int):
    """Re-express one patient's log-odds SHAP values as percentage points of risk.

    SHAP is additive on the log-odds scale, not on the probability scale, so the
    values are rescaled by the constant (p_pred - p_base) / sum(phi) to make them
    chain from the base rate to the prediction. The sigmoid is monotone, so that
    factor is always positive: every contribution keeps its SHAP sign and the
    ranking is unchanged. The rescaled numbers are an allocation of the total risk
    difference, not each predictor's marginal effect in isolation.

    Args:
        expl: Explanation from compute_shap().
        index: Row position of the patient within expl.

    Returns:
        (contributions in percentage points, base probability, predicted probability).
    """
    phi = np.asarray(expl.values[index], dtype=float)
    base = float(expl.base_values[index])
    p_base = 1.0 / (1.0 + np.exp(-base))
    p_pred = 1.0 / (1.0 + np.exp(-(base + phi.sum())))

    total = phi.sum()
    share = phi / total if total != 0 else np.zeros_like(phi)
    return share * (p_pred - p_base) * 100.0, p_base, p_pred


def _chevron(ax, x0: float, x1: float, y: float, color: str, height: float = 0.62):
    """Draw one waterfall bar from x0 to x1, pointed in its direction of travel."""
    tip = min(abs(x1 - x0) * 0.35, np.ptp(ax.get_xlim()) * 0.012)
    step = np.sign(x1 - x0) * tip
    pts = [(x0, y - height / 2), (x1 - step, y - height / 2), (x1, y),
           (x1 - step, y + height / 2), (x0, y + height / 2)]
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor=color, edgecolor="none"))


def _waterfall_xlim(rows, p_base: float, p_pred: float) -> tuple[float, float]:
    """Axis limits covering the whole chain plus a margin for the outside labels."""
    cursor, seen = p_base, [p_base, p_pred]
    for _, value in rows[::-1]:
        cursor += value / 100.0
        seen.append(cursor)
    lo, hi = min(seen), max(seen)
    pad = max((hi - lo) * 0.12, 0.02)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def _bar_label(ax, x0: float, x1: float, y: float, value: int, color: str) -> None:
    """Write the signed percentage inside the bar, or beside it when it is too thin."""
    # An unsigned "0%" for rows that round away: a signed one would contradict the
    # bar's colour whenever the sign and the rounding disagree.
    text = f"{value:+d}%" if value else "0%"
    if abs(x1 - x0) > np.ptp(ax.get_xlim()) * 0.075:
        ax.text((x0 + x1) / 2, y, text, ha="center", va="center",
                color="white", fontsize=10)
    else:
        outside = 0.006 * np.sign(x1 - x0 or 1)
        ax.text(x1 + outside, y, text, va="center", fontsize=10, color=color,
                ha="left" if outside > 0 else "right")


def _waterfall_axes(ax, rows, p_base: float, p_pred: float) -> None:
    """Apply the shared waterfall styling: tick labels, guides and spines."""
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([t for t, _ in rows][::-1], fontsize=_LABEL_SIZE)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for line in (p_base, p_pred):
        ax.axvline(line, color="0.6", linewidth=0.9, linestyle=(0, (2, 3)), zorder=0)
    ax.text(p_pred, len(rows) - 0.45, f"{p_pred:.0%}", ha="center", va="bottom",
            fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Predicted 30-day mortality risk")
    ax.grid(axis="y", color="0.92", linewidth=0.8, linestyle=(0, (1, 3)))
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _normalize_fonts(ax)


def plot_waterfall(expl, index: int, top_n: int = 15, save_path=None, label: str = ""):
    """Waterfall showing how each predictor moves one patient from base to prediction.

    Contributions are shown as whole percentage points of absolute risk, chaining
    from the cohort base rate at the bottom to this patient's prediction at the
    top. See _probability_contributions() for how the log-odds SHAP values are put
    on that scale, and note that the displayed integers are rounded to sum exactly
    to the total shift.

    Args:
        expl: Explanation from compute_shap().
        index: Row position of the patient within expl.
        top_n: Number of features to show individually (rest are pooled).
        save_path: Optional PNG destination.
        label: Optional prefix for the title (e.g. "Median-risk patient").

    Returns:
        The created figure.
    """
    pp, p_base, p_pred = _probability_contributions(expl, index)
    names = list(expl.feature_names)
    data = np.asarray(expl.data[index], dtype=float)

    order = np.argsort(-np.abs(pp))
    shown, pooled = order[:top_n], order[top_n:]
    rows = [(f"{data[i]:g} = {names[i]}", pp[i]) for i in shown]
    if len(pooled):
        rows.append((f"{len(pooled)} other features", pp[pooled].sum()))

    # Rounded for display, but the bars are drawn from the exact values so the
    # chain still lands on the prediction.
    ints = _integer_split(np.array([v for _, v in rows]),
                          int(round((p_pred - p_base) * 100)))

    fig, ax = plt.subplots(figsize=(10, 0.46 * len(rows) + 2.0))
    ax.set_xlim(*_waterfall_xlim(rows, p_base, p_pred))

    cursor = p_base
    for y, ((text, value), shown_int) in enumerate(zip(rows[::-1], ints[::-1])):
        color = _UP if value >= 0 else _DOWN
        _chevron(ax, cursor, cursor + value / 100.0, y, color)
        _bar_label(ax, cursor, cursor + value / 100.0, y, shown_int, color)
        cursor += value / 100.0

    _waterfall_axes(ax, rows, p_base, p_pred)
    # The prediction is already the bold figure above the chain, so the title only
    # has to say which patient this is.
    prefix = f"{label} — " if label else ""
    ax.set_title(f"{prefix}patient row {index}", fontsize=11, loc="left")
    _save(fig, save_path)
    return fig
