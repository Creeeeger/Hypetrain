package org.crecker;

import java.util.LinkedHashMap;
import java.util.Map;

public class weightRangeMap {
    // Trend Following Indicators (Primary predictors)
    public static final Map<String, Double> TREND_FOLLOWING_WEIGHTS = Map.ofEntries(
            Map.entry("SMA_CROSS", 0.50),
            Map.entry("MACD", 0.30),
            Map.entry("TRIX", 0.20)
    );

    // Momentum Indicators (Primary predictors)
    public static final Map<String, Double> MOMENTUM_WEIGHTS = Map.ofEntries(
            Map.entry("RSI", 0.30),
            Map.entry("ROC", 0.15),
            Map.entry("MOMENTUM", 0.25),
            Map.entry("CMO", 0.30)
    );

    // Volatility & Breakouts Indicators
    public static final Map<String, Double> VOLATILITY_BREAKOUTS_WEIGHTS = Map.ofEntries(
            Map.entry("BOLLINGER", 1.0)
    );

    // Patterns Indicators (Secondary predictors)
    public static final Map<String, Double> PATTERNS_WEIGHTS = Map.ofEntries(
            Map.entry("CONSECUTIVE_POSITIVE_CLOSES", 0.20),
            Map.entry("HIGHER_HIGHS", 0.35),
            Map.entry("TRENDLINE", 0.45)
    );

    // Statistical Indicators (Secondary predictors)
    public static final Map<String, Double> STATISTICAL_WEIGHTS = Map.ofEntries(
            Map.entry("CUMULATIVE_PERCENTAGE", 0.60),
            Map.entry("CUMULATIVE_THRESHOLD", 0.40)
    );

    // Advanced Indicators (Primary predictors)
    public static final Map<String, Double> ADVANCED_WEIGHTS = Map.ofEntries(
            Map.entry("PARABOLIC", 0.20),
            Map.entry("KELTNER", 0.35),
            Map.entry("ELDER_RAY", 0.15),
            Map.entry("ATR", 0.30)
    );

    // Aggregated weights map
    public static final Map<String, Map<String, Double>> INDICATOR_RANGE_FULL = Map.of(
            "TrendFollowing", TREND_FOLLOWING_WEIGHTS,
            "Momentum", MOMENTUM_WEIGHTS,
            "VolatilityBreakouts", VOLATILITY_BREAKOUTS_WEIGHTS,
            "Patterns", PATTERNS_WEIGHTS,
            "Statistical", STATISTICAL_WEIGHTS,
            "Advanced", ADVANCED_WEIGHTS
    );

    // Category Level Weights (Sum = 1.0)
    public static final Map<String, Double> INDICATOR_WEIGHTS_FULL = Map.of(
            "TrendFollowing", 0.25,  // Strong trend signals
            "Momentum", 0.25,       // Momentum confirmation
            "VolatilityBreakouts", 0.15,  // Breakout detection
            "Patterns", 0.10,       // Subjective patterns
            "Statistical", 0.10,    // Long-term metrics
            "Advanced", 0.15        // Hybrid signals
    );

    public static final Map<String, Map<String, Double>> INDICATOR_RANGE_MAP = new LinkedHashMap<>() {{
        // Trend Following Indicators
        put("SMA_CROSS", Map.of("min", -1.0, "max", 1.0));
        put("MACD", Map.of("min", -1.0, "max", 1.0));
        put("TRIX", Map.of("min", -1.0, "max", 1.0));

        // Momentum Indicators
        put("RSI", Map.of("min", 0.0, "max", 100.0));
        put("ROC", Map.of("min", -5.0, "max", 5.0));
        put("MOMENTUM", Map.of("min", -1.0, "max", 1.0));
        put("CMO", Map.of("min", -100.0, "max", 100.0));

        // Volatility & Breakouts Indicators
        put("BOLLINGER", Map.of("min", 0.0, "max", 0.5));

        // Patterns Indicators
        put("CONSECUTIVE_POSITIVE_CLOSES", Map.of("min", 0.0, "max", 50.0));
        put("HIGHER_HIGHS", Map.of("min", 0.0, "max", 1.0));
        put("TRENDLINE", Map.of("min", 0.0, "max", 1.0));

        // Statistical Indicators
        put("CUMULATIVE_PERCENTAGE", Map.of("min", 0.0, "max", 1.0));
        put("CUMULATIVE_THRESHOLD", Map.of("min", -10.0, "max", 10.0));

        // Advanced Indicators
        put("PARABOLIC", Map.of("min", 0.0, "max", 1.0));
        put("KELTNER", Map.of("min", 0.0, "max", 1.0));
        put("ELDER_RAY", Map.of("min", -1.0, "max", 1.0));
        put("ATR", Map.of("min", 0.0, "max", 1.0));
    }};
}