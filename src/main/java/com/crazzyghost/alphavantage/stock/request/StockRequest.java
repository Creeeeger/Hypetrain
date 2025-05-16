package com.crazzyghost.alphavantage.stock.request;

import com.crazzyghost.alphavantage.parameters.Function;

/**
 * Represents a request for stock-related information using the Alpha Vantage API.
 * <p>
 * Encapsulates the API function (such as SYMBOL_SEARCH) and any associated keywords.
 * Instances of this class are typically constructed using the {@link Builder} inner class.
 */
public class StockRequest {

    /**
     * The Alpha Vantage API function to call (e.g., SYMBOL_SEARCH).
     */
    protected Function function;

    /**
     * Search keywords (such as a company name or symbol) used for the request.
     */
    protected String keywords;

    /**
     * Constructs a StockRequest object using the given builder instance.
     *
     * @param builder Builder instance holding request parameters.
     */
    protected StockRequest(Builder builder) {
        this.keywords = builder.keywords;
        this.function = builder.function;
    }

    /**
     * Builder class for constructing {@link StockRequest} objects in a flexible and readable way.
     * <p>
     * Allows the user to set function type and search keywords, and then build the final immutable request.
     */
    public static class Builder {

        /**
         * The Alpha Vantage API function to use (default is SYMBOL_SEARCH).
         */
        public Function function;

        /**
         * Search keywords for the stock request.
         */
        protected String keywords;

        /**
         * Constructs a new Builder with default settings.
         * <p>
         * By default, the function is set to {@link Function#SYMBOL_SEARCH}.
         */
        public Builder() {
            this.function = Function.SYMBOL_SEARCH;
        }

        /**
         * Sets the function type for the stock request.
         *
         * @param function Alpha Vantage function (e.g., TIME_SERIES_DAILY).
         * @return This Builder instance for method chaining.
         */
        public Builder function(Function function) {
            this.function = function;
            return this;
        }

        /**
         * Sets the search keywords for the stock request.
         *
         * @param keywords Search string (e.g., company name, ticker symbol).
         */
        public void forKeywords(String keywords) {
            this.keywords = keywords;
        }

        /**
         * Builds and returns a new {@link StockRequest} instance using the current builder settings.
         *
         * @return New immutable StockRequest object.
         */
        public StockRequest build() {
            return new StockRequest(this); // Use the constructor to create an instance
        }
    }
}