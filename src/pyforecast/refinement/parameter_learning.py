"""Parameter learning from accumulated fit logs.

Analyzes historical fit results to suggest optimal fitting parameters
for different groupings (global, by basin, by formation).
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np

from .schemas import FitLogRecord, ParameterSuggestion
from .storage import FitLogStorage

logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """Configuration for parameter learning.

    Attributes:
        min_samples: Minimum samples required for suggestion
        min_samples_high_confidence: Samples for high confidence
        target_mape: Target MAPE for parameter optimization
        recency_half_life_range: Range to search for half-life
        regime_threshold_range: Range to search for threshold
    """

    min_samples: int = 10
    min_samples_high_confidence: int = 100
    target_mape: float = 20.0
    recency_half_life_range: tuple[float, float] = (6.0, 24.0)
    regime_threshold_range: tuple[float, float] = (0.5, 2.0)


class ParameterLearner:
    """Learns optimal fitting parameters from historical fit logs.

    Analyzes accumulated fit logs to identify which parameter combinations
    produce the best hindcast performance for different groupings.

    Example:
        learner = ParameterLearner()

        # Get suggestions for a basin/formation
        suggestion = learner.suggest(
            product="oil",
            basin="Permian",
            formation="Wolfcamp"
        )

        if suggestion:
            print(f"Suggested half-life: {suggestion.suggested_recency_half_life}")
            print(f"Based on {suggestion.sample_count} fits")
            print(f"Average MAPE: {suggestion.avg_hindcast_mape:.1f}%")
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
        config: LearningConfig | None = None,
    ):
        """Initialize parameter learner.

        Args:
            storage_path: Path to fit log storage (None = default)
            config: Learning configuration
        """
        self.storage = FitLogStorage(storage_path)
        self.config = config or LearningConfig()

    def suggest(
        self,
        product: str,
        basin: str | None = None,
        formation: str | None = None,
    ) -> ParameterSuggestion | None:
        """Get parameter suggestion for a grouping.

        Tries to find suggestions at the most specific level available,
        falling back to less specific groupings.

        Args:
            product: Product type (oil, gas, water)
            basin: Basin name (optional)
            formation: Formation name (optional)

        Returns:
            ParameterSuggestion if sufficient data, None otherwise
        """
        # Try stored suggestion first
        stored = self.storage.get_suggestion(product, basin, formation)
        if stored is not None:
            return stored

        # Compute new suggestion
        return self._compute_suggestion(product, basin, formation)

    def _compute_suggestion(
        self,
        product: str,
        basin: str | None = None,
        formation: str | None = None,
    ) -> ParameterSuggestion | None:
        """Compute parameter suggestion from fit logs.

        Args:
            product: Product type
            basin: Basin name (optional)
            formation: Formation name (optional)

        Returns:
            ParameterSuggestion if sufficient data, None otherwise
        """
        # Try specific first, fall back to less specific
        for b, f in [(basin, formation), (basin, None), (None, None)]:
            records = self.storage.query(product=product, basin=b, formation=f)

            # Filter to records with hindcast data
            with_hindcast = [r for r in records if r.hindcast_mape is not None]

            if len(with_hindcast) >= self.config.min_samples:
                return self._analyze_records(with_hindcast, product, b, f)

        return None

    def _analyze_records(
        self,
        records: list[FitLogRecord],
        product: str,
        basin: str | None,
        formation: str | None,
    ) -> ParameterSuggestion:
        """Analyze records to derive parameter suggestions.

        Uses a weighted analysis approach:
        - Group records by parameter values
        - Weight by hindcast performance (lower MAPE = higher weight)
        - Derive optimal values

        Args:
            records: Fit log records with hindcast data
            product: Product type
            basin: Basin name
            formation: Formation name

        Returns:
            ParameterSuggestion with derived values
        """
        # Extract parameter values and performance
        half_lives = np.array([r.recency_half_life for r in records])
        thresholds = np.array([r.regime_threshold for r in records])
        mapes = np.array([r.hindcast_mape for r in records])
        r_squareds = np.array([r.r_squared for r in records])

        # Compute weights based on performance (inverse MAPE)
        # Clip MAPE to avoid extreme weights
        mapes_clipped = np.clip(mapes, 5.0, 100.0)
        weights = 1.0 / mapes_clipped
        weights = weights / np.sum(weights)

        # Compute weighted averages
        suggested_half_life = float(np.sum(half_lives * weights))
        suggested_threshold = float(np.sum(thresholds * weights))

        # Clip to valid ranges
        suggested_half_life = np.clip(
            suggested_half_life,
            self.config.recency_half_life_range[0],
            self.config.recency_half_life_range[1],
        )
        suggested_threshold = np.clip(
            suggested_threshold,
            self.config.regime_threshold_range[0],
            self.config.regime_threshold_range[1],
        )

        suggested_window = 6
        suggested_sustained = 2

        # Compute performance metrics
        avg_r_squared = float(np.mean(r_squareds))
        avg_mape = float(np.mean(mapes))

        # Build grouping string
        if basin:
            grouping = basin
            if formation:
                grouping = f"{basin}/{formation}"
        else:
            grouping = "global"

        suggestion = ParameterSuggestion(
            grouping=grouping,
            sample_count=len(records),
            product=product,
            suggested_recency_half_life=suggested_half_life,
            suggested_regime_threshold=suggested_threshold,
            suggested_regime_window=suggested_window,
            suggested_regime_sustained_months=suggested_sustained,
            avg_r_squared=avg_r_squared,
            avg_hindcast_mape=avg_mape,
        )

        return suggestion

    def update_suggestions(self, product: str | None = None) -> int:
        """Update all parameter suggestions from current fit logs.

        Recomputes suggestions for all groupings found in the database.

        Args:
            product: If specified, only update for this product

        Returns:
            Number of suggestions updated
        """
        products = [product] if product else ["oil", "gas", "water"]
        updated = 0

        for prod in products:
            # Get unique basin/formation combinations
            groupings = self._get_groupings(prod)

            for basin, formation in groupings:
                suggestion = self._compute_suggestion(prod, basin, formation)
                if suggestion is not None:
                    self.storage.save_suggestion(suggestion)
                    updated += 1
                    logger.debug(
                        f"Updated suggestion for {prod}/{basin or 'global'}"
                        f"/{formation or 'all'}"
                    )

        logger.info(f"Updated {updated} parameter suggestions")
        return updated

    def _get_groupings(self, product: str) -> list[tuple[str | None, str | None]]:
        """Get unique basin/formation groupings from database.

        Args:
            product: Product to query

        Returns:
            List of (basin, formation) tuples
        """
        import sqlite3

        groupings = [(None, None)]  # Always include global

        with sqlite3.connect(self.storage.db_path) as conn:
            # Get unique basins
            cursor = conn.execute(
                "SELECT DISTINCT basin FROM fit_logs WHERE product = ? AND basin IS NOT NULL",
                (product,)
            )
            basins = [row[0] for row in cursor.fetchall()]

            for basin in basins:
                groupings.append((basin, None))

                # Get formations within basin
                cursor = conn.execute(
                    "SELECT DISTINCT formation FROM fit_logs "
                    "WHERE product = ? AND basin = ? AND formation IS NOT NULL",
                    (product, basin)
                )
                formations = [row[0] for row in cursor.fetchall()]

                for formation in formations:
                    groupings.append((basin, formation))

        return groupings

    def get_all_suggestions(self, product: str | None = None) -> list[ParameterSuggestion]:
        """Get all stored parameter suggestions.

        Args:
            product: If specified, filter to this product

        Returns:
            List of ParameterSuggestion objects
        """
        import sqlite3

        sql = "SELECT * FROM parameter_suggestions"
        params = []
        if product:
            sql += " WHERE product = ?"
            params.append(product)

        suggestions = []
        with sqlite3.connect(self.storage.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                grouping = "global"
                if row["basin"]:
                    grouping = row["basin"]
                    if row["formation"]:
                        grouping = f"{row['basin']}/{row['formation']}"

                suggestions.append(ParameterSuggestion(
                    grouping=grouping,
                    sample_count=row["sample_count"],
                    product=row["product"],
                    suggested_recency_half_life=row["suggested_recency_half_life"],
                    suggested_regime_threshold=row["suggested_regime_threshold"],
                    suggested_regime_window=row["suggested_regime_window"],
                    suggested_regime_sustained_months=row["suggested_regime_sustained_months"],
                    avg_r_squared=row["avg_r_squared"],
                    avg_hindcast_mape=row["avg_hindcast_mape"],
                ))

        return suggestions

    def export_suggestions(self, output_path: Path | str) -> int:
        """Export all suggestions to CSV.

        Args:
            output_path: Path for output CSV

        Returns:
            Number of suggestions exported
        """
        import csv

        output_path = Path(output_path)
        suggestions = self.get_all_suggestions()

        if not suggestions:
            return 0

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "grouping",
                "product",
                "sample_count",
                "confidence",
                "suggested_recency_half_life",
                "suggested_regime_threshold",
                "suggested_regime_window",
                "suggested_regime_sustained_months",
                "avg_r_squared",
                "avg_hindcast_mape",
            ])

            for s in suggestions:
                writer.writerow([
                    s.grouping,
                    s.product,
                    s.sample_count,
                    s.confidence,
                    s.suggested_recency_half_life,
                    s.suggested_regime_threshold,
                    s.suggested_regime_window,
                    s.suggested_regime_sustained_months,
                    s.avg_r_squared,
                    s.avg_hindcast_mape,
                ])

        return len(suggestions)


def apply_suggestion(
    suggestion: ParameterSuggestion,
    fitting_config: "FittingConfig",  # noqa: F821
) -> "FittingConfig":  # noqa: F821
    """Create a new FittingConfig with suggested parameters applied.

    Note: This returns a new config - it does not modify the original.
    The caller decides whether to use the suggested config.

    Args:
        suggestion: ParameterSuggestion to apply
        fitting_config: Base FittingConfig to modify

    Returns:
        New FittingConfig with suggested parameters
    """
    from ..core.fitting import FittingConfig

    return FittingConfig(
        b_min=fitting_config.b_min,
        b_max=fitting_config.b_max,
        dmin_annual=fitting_config.dmin_annual,
        regime_threshold=suggestion.suggested_regime_threshold,
        regime_window=suggestion.suggested_regime_window,
        regime_sustained_months=suggestion.suggested_regime_sustained_months,
        recency_half_life=suggestion.suggested_recency_half_life,
        min_points=fitting_config.min_points,
    )
