package com.crazzyghost.alphavantage.stock.request;

import com.crazzyghost.alphavantage.parameters.Function;

public class StockRequest {

    protected Function function;
    protected String keywords;

    protected StockRequest(Builder builder) {
        this.keywords = builder.keywords;
        this.function = builder.function;
    }

    public static class Builder {

        public Function function;
        protected String keywords;

        public Builder() {
            this.function = Function.SYMBOL_SEARCH;
        }

        public Builder function(Function function) {
            this.function = function;
            return this;
        }

        public Builder forKeywords(String keywords) {
            this.keywords = keywords;
            return this;
        }

        public StockRequest build() {
            return new StockRequest(this); // Use the constructor to create an instance
        }
    }
}