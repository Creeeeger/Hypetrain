package com.crazzyghost.alphavantage.realtime.request;

import com.crazzyghost.alphavantage.parameters.Function;

/**
 * Represents a request for real-time bulk quotes from the Alpha Vantage API.
 * Supports configuration of the function type and the symbols to query.
 * Utilizes the Builder pattern for flexible and readable construction.
 */
public class RealTimeRequest {

    /**
     * The Alpha Vantage function specifying the API endpoint, e.g. REALTIME_BULK_QUOTES
     */
    private final Function function;

    /**
     * Comma-separated list of stock symbols for which to fetch real-time quotes
     */
    private final String symbol;

    // realtime or delayed
    private final String entitlement;

    /**
     * Private constructor called by the Builder.
     *
     * @param builder Builder instance containing configuration parameters
     */
    private RealTimeRequest(Builder builder) {
        this.function = builder.function;
        this.symbol = builder.symbol;
        this.entitlement = builder.entitlement;
    }

    /**
     * Gets the API function configured for this request.
     *
     * @return the function enum value
     */
    public Function getFunction() {
        return function;
    }

    /**
     * Gets the comma-separated stock symbols for this request.
     *
     * @return symbols string (e.g., "AAPL,MSFT,GOOG")
     */
    public String getSymbol() {
        return symbol;
    }

    public String getEntitlement() {
        return entitlement;
    }

    /**
     * Builder class to create instances of RealTimeRequest with flexible parameters.
     */
    public static class Builder {

        /**
         * Function to specify API endpoint, defaults to REALTIME_BULK_QUOTES
         */
        private Function function;

        /**
         * Comma-separated stock symbols to query
         */
        private String symbol;

        // realtime or delayed
        private String entitlement;

        /**
         * Initializes the Builder with default function REALTIME_BULK_QUOTES.
         */
        public Builder() {
            this.function = Function.REALTIME_BULK_QUOTES;
        }

        /**
         * Sets the Alpha Vantage function for the request.
         *
         * @param function API function enum value
         * @return this Builder for chaining
         */
        public Builder function(Function function) {
            this.function = function;
            return this;
        }

        /**
         * Sets the stock symbol to query.
         *
         * @param symbol comma-separated stock symbol string
         * @return this Builder for chaining
         */
        public Builder symbol(String symbol) {
            this.symbol = symbol;
            return this;
        }

        // set realtime or delayed
        public Builder entitlement(String entitlement) {
            this.entitlement = entitlement;
            return this;
        }

        /**
         * Builds the RealTimeRequest instance with the configured parameters.
         *
         * @return a new RealTimeRequest object
         */
        public RealTimeRequest build() {
            return new RealTimeRequest(this);
        }
    }
}