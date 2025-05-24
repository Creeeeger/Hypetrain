package com.crazzyghost.alphavantage.alphaIntelligence.request;

import static com.crazzyghost.alphavantage.parameters.Function.INSIDER_TRANSACTIONS;

/**
 * Represents a request for the Alpha Vantage Insider Transactions endpoint.
 * <p>
 * This endpoint retrieves recent insider transaction data for a specified stock symbol.
 *
 * <h2>Usage Example:</h2>
 * <pre>
 * InsiderTransactionsRequest request = new InsiderTransactionsRequest.Builder()
 *     .symbol("AAPL")
 *     .build();
 * </pre>
 *
 * @see AlphaIntelligenceRequest
 */
public class InsiderTransactionsRequest extends AlphaIntelligenceRequest {
    /**
     * The stock symbol to retrieve insider transaction data for.
     */
    private final String symbol;

    /**
     * Private constructor. Use {@link Builder} to construct an instance.
     *
     * @param builder The builder containing configuration values.
     */
    private InsiderTransactionsRequest(Builder builder) {
        this.function = builder.function;
        this.symbol = builder.symbol;
    }

    /**
     * Gets the stock symbol for which insider transactions will be requested.
     *
     * @return the stock symbol as a String (e.g., "AAPL").
     */
    public String getSymbol() {
        return symbol;
    }

    /**
     * Builder for {@link InsiderTransactionsRequest}.
     * <p>
     * Provides a fluent API for configuring and creating an insider transactions request.
     */
    public static class Builder extends AlphaIntelligenceRequest.Builder<Builder> {
        /**
         * The stock symbol to query.
         */
        private String symbol;

        /**
         * Constructs a new Builder with the function set to INSIDER_TRANSACTIONS.
         */
        public Builder() {
            this.function = INSIDER_TRANSACTIONS;
        }

        /**
         * Sets the stock symbol to retrieve insider transaction data for.
         *
         * @param symbol the ticker symbol (e.g., "AAPL", "MSFT")
         * @return this builder instance for fluent chaining
         */
        public Builder symbol(String symbol) {
            this.symbol = symbol;
            return this;
        }

        /**
         * Builds and returns a new {@link InsiderTransactionsRequest} using
         * the parameters set in this builder.
         *
         * @return a fully configured InsiderTransactionsRequest
         */
        @Override
        public InsiderTransactionsRequest build() {
            return new InsiderTransactionsRequest(this);
        }
    }
}