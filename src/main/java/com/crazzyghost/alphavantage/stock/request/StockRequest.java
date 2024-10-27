package com.crazzyghost.alphavantage.stock.request;

import com.crazzyghost.alphavantage.parameters.Function;

public class StockRequest {

    protected Function function;
    protected String symbol;
    protected int volume;

    protected StockRequest(Builder builder) {
        this.symbol = builder.symbol;
        this.function = builder.function;
        this.volume = builder.volume;
    }

    public static class Builder {

        public Function function;
        protected String symbol;
        protected int volume;

        public Builder() {
            this.function = Function.SYMBOL_SEARCH;
        }

        public Builder function(Function function) {
            this.function = function;
            return this;
        }

        public Builder forSymbol(String symbol) {
            this.symbol = symbol;
            return this;
        }

        public Builder forVolume(int volume) {
            this.volume = volume;
            return this;
        }

        public StockRequest build() {
            return new StockRequest(this); // Use the constructor to create an instance
        }
    }
}