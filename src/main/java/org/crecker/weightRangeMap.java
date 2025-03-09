package org.crecker;

import java.util.LinkedHashMap;
import java.util.Map;

public class weightRangeMap {
    // Trend Following Indicators
    public static final Map<String, Double> TREND_FOLLOWING_WEIGHTS = Map.ofEntries(
            Map.entry("SMA_CROSS", 0.5),
            Map.entry("TRIX", 0.5)
    );

    // Momentum Indicators
    public static final Map<String, Double> MOMENTUM_WEIGHTS = Map.ofEntries(
            Map.entry("ROC", 1.0)
    );

    // Volatility & Breakouts Indicators
    public static final Map<String, Double> VOLATILITY_BREAKOUTS_WEIGHTS = Map.ofEntries(
            Map.entry("BOLLINGER", 1.0)
    );

    // Statistical Indicators
    public static final Map<String, Double> STATISTICAL_WEIGHTS = Map.ofEntries(
            Map.entry("CUMULATIVE_PERCENTAGE", 0.6),
            Map.entry("CUMULATIVE_THRESHOLD", 0.4)
    );

    // Advanced Indicators
    public static final Map<String, Double> ADVANCED_WEIGHTS = Map.ofEntries(
            Map.entry("KELTNER", 0.4),
            Map.entry("ELDER_RAY", 0.6)
    );

    // Aggregated weights map
    public static final Map<String, Map<String, Double>> INDICATOR_RANGE_FULL = Map.of(
            "TrendFollowing", TREND_FOLLOWING_WEIGHTS,
            "Momentum", MOMENTUM_WEIGHTS,
            "VolatilityBreakouts", VOLATILITY_BREAKOUTS_WEIGHTS,
            "Statistical", STATISTICAL_WEIGHTS,
            "Advanced", ADVANCED_WEIGHTS
    );

    // Category Level Weights (Sum = 1.0)
    public static final Map<String, Double> INDICATOR_WEIGHTS_FULL = Map.of(
            "TrendFollowing", 0.3,
            "Momentum", 0.1,
            "VolatilityBreakouts", 0.1,
            "Statistical", 0.2,
            "Advanced", 0.3
    );

    public static final Map<String, Map<String, Double>> INDICATOR_RANGE_MAP = new LinkedHashMap<>() {{
        // Trend Following Indicators
        put("SMA_CROSS", Map.of("min", -1.0, "max", 1.0));
        put("TRIX", Map.of("min", -0.5, "max", 0.5));

        // Momentum Indicators
        put("ROC", Map.of("min", -5.0, "max", 5.0));

        // Volatility & Breakouts Indicators
        put("BOLLINGER", Map.of("min", 0.0, "max", 0.1));

        // Statistical Indicators
        put("CUMULATIVE_PERCENTAGE", Map.of("min", 0.0, "max", 1.0));
        put("CUMULATIVE_THRESHOLD", Map.of("min", -7.0, "max", 7.0));

        // Advanced Indicators
        put("KELTNER", Map.of("min", 0.0, "max", 1.0));
        put("ELDER_RAY", Map.of("min", -3.0, "max", 3.0));
    }};
}