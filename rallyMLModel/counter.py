# Define the weights in a list
weights = [
    0.08, 0.09, 0.07, 0.06, 0.05, 0.04, 0.05, 0.05, 0.04, 0.03,
    0.04, 0.03, 0.03, 0.03, 0.03, 0.01, 0.01, 0.03, 0.02, 0.03,
    0.01, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02
]

# Calculate the sum of the weights
total_weight = sum(weights)

# Print the total sum
print(f"Total weight: {total_weight}")

# Let's count the number of entries in the INDICATOR_RANGES map.
indicator_ranges = {
    "RSI": {"min": 0.0, "max": 100.0},
    "MACD": {"min": -2.0, "max": 2.0},
    "BOLLINGER": {"min": -100.0, "max": 100.0},
    "VOLUME": {"min": 0.0, "max": 5.0},
    "TRIX": {"min": -1.0, "max": 1.0},
    "KAMA": {"min": -10.0, "max": 10.0},
    "CMO": {"min": -100.0, "max": 100.0},
    "ACCELERATION": {"min": -5.0, "max": 5.0},
    "DONCHIAN": {"min": 0.0, "max": 1.0},
    "Z_SCORE": {"min": -3.0, "max": 3.0},
    "CUMULATIVE": {"min": -20.0, "max": 20.0},
    "KELTNER": {"min": 0.0, "max": 1.0},
    "ELDER_RAY": {"min": -10.0, "max": 10.0},
    "PARABOLIC": {"min": 0.0, "max": 1.0},
    "TRENDLINE": {"min": 0.0, "max": 1.0},
    "EMA_CROSS": {"min": 0.0, "max": 100.0},
    "ROC": {"min": -100.0, "max": 100.0},
    "MOMENTUM": {"min": -100.0, "max": 100.0},
    "BOLLINGER_BANDWIDTH": {"min": 0.0, "max": 1.0},
    "SMA_CROSS": {"min": 0.0, "max": 1.0},
    "PRICE_SMA_DISTANCE": {"min": -10.0, "max": 10.0},
    "FRACTAL_BREAKOUT": {"min": 0.0, "max": 1.0},
    "HIGHER_HIGHS": {"min": 0.0, "max": 1.0},
    "BREAKOUT_MA": {"min": 0.0, "max": 1.0},
    "VOLUME_SPIKE": {"min": 0.0, "max": 1.0},
    "CUMULATIVE_PERCENTAGE": {"min": 0.0, "max": 1.0},
    "BREAKOUT_RESISTANCE": {"min": 0.0, "max": 1.0},
    "VOLATILITY_THRESHOLD": {"min": 0.0, "max": 1.0},
    "VOLATILITY_MONITOR": {"min": 0.0, "max": 1.0},
    "CANDLE_PATTERN": {"min": 0.0, "max": 1.0},
    "ATR": {"min": 0.0, "max": 1.0},
    "CONSECUTIVE_POSITIVE_CLOSES": {"min": 0.0, "max": 1.0},
    "CUMULATIVE_THRESHOLD": {"min": 0.0, "max": 1.0}
}

# Counting the entries in the indicator_ranges dictionary
print(len(indicator_ranges))

features = [
    'sma_crossover', 'ema_crossover', 'price_above_sma', 'macd_line', 'trix', 'kama',
    'rsi', 'roc', 'momentum', 'cmo', 'acceleration',
    'bollinger_bands', 'breakout', 'donchian_breakout', 'volatility_spike', 'volatility_ratio', 'positive_closes',
    'higher_highs', 'fractal_breakout', 'candle_patterns', 'trendline_breakout', 'zscore_spike', 'cumulative_spike',
    'cumulative_change', 'breakout_above_ma', 'parabolic_sar_bullish', 'keltner_breakout', 'elder_ray_index',
    'volume_spike', 'atr'
]

print(len(features))
