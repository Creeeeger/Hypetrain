package com.crazzyghost.alphavantage.alphaIntelligence.request;

import static com.crazzyghost.alphavantage.parameters.Function.ANALYTICS_FIXED_WINDOW;

/**
 * Represents a request for the Alpha Vantage Analytics Fixed Window endpoint.
 * <p>
 * This endpoint provides analytics over a fixed time window for one or more symbols,
 * supporting configurable range, interval, calculation type, and OHLC inclusion.
 *
 * <h2>Usage Example:</h2>
 * <pre>
 *     AnalyticsFixedWindowRequest request = new AnalyticsFixedWindowRequest.Builder()
 *         .symbols("AAPL,MSFT")
 *         .range("1d", "5d")
 *         .interval("5min")
 *         .calculations("mean,median")
 *         .ohlc("true")
 *         .build();
 * </pre>
 */
public class AnalyticsFixedWindowRequest extends AlphaIntelligenceRequest {
    /**
     * Comma-separated list of symbol(s) to query (e.g., "AAPL,MSFT").
     */
    private final String symbols;
    /**
     * String of range value
     */
    private final String range;
    /**
     * Data interval granularity (e.g., "1min", "5min", "1h").
     */
    private final String interval;
    /**
     * Calculations to perform, as a comma-separated string (e.g., "mean,median").
     */
    private final String calculations;
    /**
     * Whether to include OHLC (open, high, low, close) data ("true" or "false").
     */
    private final String ohlc;

    /**
     * Private constructor. Use the Builder to construct an instance.
     *
     * @param builder The builder holding all parameter values.
     */
    private AnalyticsFixedWindowRequest(Builder builder) {
        this.function = builder.function;
        this.symbols = builder.symbols;
        this.range = builder.range;
        this.interval = builder.interval;
        this.calculations = builder.calculations;
        this.ohlc = builder.ohlc;
    }

    /**
     * @return the symbols specified for this request.
     */
    public String getSymbols() {
        return symbols;
    }

    /**
     * @return the range value for the time window.
     */
    public String getRange() {
        return range;
    }

    /**
     * @return the data interval granularity.
     */
    public String getInterval() {
        return interval;
    }

    /**
     * @return the requested calculation types as a string.
     */
    public String getCalculations() {
        return calculations;
    }

    /**
     * @return whether OHLC data is requested ("true" or "false").
     */
    public String getOhlc() {
        return ohlc;
    }

    /**
     * Builder for {@link AnalyticsFixedWindowRequest}.
     * <p>
     * Provides a fluent API for constructing an analytics request.
     */
    public static class Builder extends AlphaIntelligenceRequest.Builder<Builder> {
        private String symbols;
        private String range;
        private String interval;
        private String calculations;
        private String ohlc;

        /**
         * Constructs a new Builder with the function set to ANALYTICS_FIXED_WINDOW.
         */
        public Builder() {
            this.function = ANALYTICS_FIXED_WINDOW;
        }

        /**
         * Set the symbols for the request.
         *
         * @param symbols Comma-separated string of stock/asset symbols.
         * @return this builder instance
         */
        public Builder symbols(String symbols) {
            this.symbols = symbols;
            return this;
        }

        /**
         * Set the range for the time window.
         *
         * @param range string range.
         * @return this builder instance
         */
        public Builder range(String range) {
            this.range = range;
            return this;
        }

        /**
         * Set the data interval.
         *
         * @param interval Interval string (e.g., "5min", "1h").
         * @return this builder instance
         */
        public Builder interval(String interval) {
            this.interval = interval;
            return this;
        }

        /**
         * Set the calculations to perform.
         *
         * @param calculations Comma-separated calculations (e.g., "mean,median").
         * @return this builder instance
         */
        public Builder calculations(String calculations) {
            this.calculations = calculations;
            return this;
        }

        /**
         * Specify whether to include OHLC data in the result.
         *
         * @param ohlc "true" to include OHLC, "false" otherwise.
         * @return this builder instance
         */
        public Builder ohlc(String ohlc) {
            this.ohlc = ohlc;
            return this;
        }

        /**
         * Build and return an {@link AnalyticsFixedWindowRequest} instance
         * using the parameters configured in this builder.
         *
         * @return new AnalyticsFixedWindowRequest object
         */
        @Override
        public AnalyticsFixedWindowRequest build() {
            return new AnalyticsFixedWindowRequest(this);
        }
    }
}