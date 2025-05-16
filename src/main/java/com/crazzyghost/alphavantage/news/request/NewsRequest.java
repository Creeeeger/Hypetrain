package com.crazzyghost.alphavantage.news.request;

import com.crazzyghost.alphavantage.parameters.Function;

/**
 * Represents a request to fetch news data from the Alpha Vantage API.
 * This request can be customized by filtering news based on tickers, topics,
 * date ranges, sorting options, and the maximum number of results returned.
 * <p>
 * The class uses the Builder pattern to allow flexible and readable construction of requests.
 */
public class NewsRequest {

    /**
     * The Alpha Vantage function to specify the API endpoint. Defaults to NEWS_SENTIMENT.
     */
    protected Function function;

    /**
     * Optional: Comma-separated list of stock tickers to filter news.
     */
    protected String tickers;

    /**
     * Optional: Comma-separated list of topics to filter news.
     */
    protected String topics;

    /**
     * Optional: Start datetime (ISO 8601 format) to filter news from this time onwards.
     */
    protected String timeFrom;

    /**
     * Optional: End datetime (ISO 8601 format) to filter news until this time.
     */
    protected String timeTo;

    /**
     * Optional: Sorting criteria for the results (e.g., "LATEST", "RELEVANT").
     */
    protected String sort;

    /**
     * Optional: Maximum number of news results to return. Defaults to 50.
     */
    protected int limit;

    /**
     * Private constructor called by the Builder to instantiate a NewsRequest.
     *
     * @param builder the Builder containing all parameters for the request
     */
    protected NewsRequest(Builder builder) {
        this.function = builder.function;
        this.tickers = builder.tickers;
        this.topics = builder.topics;
        this.timeFrom = builder.timeFrom;
        this.timeTo = builder.timeTo;
        this.sort = builder.sort;
        this.limit = builder.limit;
    }

    /**
     * Builder class to construct instances of NewsRequest with flexible parameters.
     */
    public static class Builder {

        /**
         * The Alpha Vantage function (API endpoint) to call. Default is NEWS_SENTIMENT.
         */
        public Function function;

        /**
         * Optional: Comma-separated tickers to filter the news.
         */
        protected String tickers;

        /**
         * Optional: Comma-separated topics to filter the news.
         */
        protected String topics;

        /**
         * Optional: Starting datetime (ISO 8601) for filtering news.
         */
        protected String timeFrom;

        /**
         * Optional: Ending datetime (ISO 8601) for filtering news.
         */
        protected String timeTo;

        /**
         * Optional: Sort order of the results.
         */
        protected String sort;

        /**
         * Optional: Limit on the number of results. Defaults to 50.
         */
        protected int limit = 50;

        /**
         * Creates a new Builder instance with default function NEWS_SENTIMENT.
         */
        public Builder() {
            this.function = Function.NEWS_SENTIMENT;
        }

        /**
         * Sets the Alpha Vantage function for the request.
         *
         * @param function the API function to use
         * @return this Builder for chaining
         */
        public Builder function(Function function) {
            this.function = function;
            return this;
        }

        /**
         * Sets the tickers to filter news results.
         *
         * @param tickers comma-separated ticker symbols (e.g. "AAPL,MSFT")
         * @return this Builder for chaining
         */
        public Builder tickers(String tickers) {
            this.tickers = tickers;
            return this;
        }

        /**
         * Sets the topics to filter news results.
         *
         * @param topics comma-separated topic keywords (e.g. "technology,finance")
         * @return this Builder for chaining
         */
        public Builder topics(String topics) {
            this.topics = topics;
            return this;
        }

        /**
         * Sets the start datetime for news filtering.
         *
         * @param timeFrom ISO 8601 formatted datetime string (e.g. "2023-01-01T00:00:00Z")
         * @return this Builder for chaining
         */
        public Builder timeFrom(String timeFrom) {
            this.timeFrom = timeFrom;
            return this;
        }

        /**
         * Sets the end datetime for news filtering.
         *
         * @param timeTo ISO 8601 formatted datetime string (e.g. "2023-01-31T23:59:59Z")
         * @return this Builder for chaining
         */
        public Builder timeTo(String timeTo) {
            this.timeTo = timeTo;
            return this;
        }

        /**
         * Sets the sorting order for the news results.
         *
         * @param sort sorting option such as "LATEST" or "RELEVANT"
         * @return this Builder for chaining
         */
        public Builder sort(String sort) {
            this.sort = sort;
            return this;
        }

        /**
         * Sets the maximum number of news results to return.
         *
         * @param limit the maximum number of results (must be positive)
         * @return this Builder for chaining
         */
        public Builder limit(int limit) {
            this.limit = limit;
            return this;
        }

        /**
         * Builds the NewsRequest instance with the configured parameters.
         *
         * @return a new NewsRequest object
         */
        public NewsRequest build() {
            return new NewsRequest(this);  // Use constructor to create instance
        }
    }
}