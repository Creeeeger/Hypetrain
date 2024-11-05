package com.crazzyghost.alphavantage.realtime.request;

import com.crazzyghost.alphavantage.parameters.Function;

public class RealTimeRequest {

    private final Function function;
    private final String symbols;

    private RealTimeRequest(Builder builder) {
        this.function = builder.function;
        this.symbols = builder.symbols;
    }

    public Function getFunction() {
        return function;
    }

    public String getSymbols() {
        return symbols;
    }

    public static class Builder {

        private Function function;
        private String symbols;

        public Builder() {
            this.function = Function.REALTIME_BULK_QUOTES; // Default function for real-time bulk quotes
        }

        public Builder function(Function function) {
            this.function = function;
            return this;
        }

        public Builder symbols(String symbols) {
            this.symbols = symbols;
            return this;
        }

        public RealTimeRequest build() {
            return new RealTimeRequest(this);
        }
    }
}