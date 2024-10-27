package com.crazzyghost.alphavantage.stock.request;

import com.crazzyghost.alphavantage.parameters.Function;

import java.util.ArrayList;
import java.util.List;

public class VolumeRequest extends StockRequest {
    private final List<VolumeRequest> results;

    protected VolumeRequest(Builder builder) {
        super(builder);
        this.results = Builder.results;
    }

    public static class Builder extends StockRequest.Builder {
        static List<VolumeRequest> results = new ArrayList<>();

        public Builder() {
            super();
            this.function(Function.SYMBOL_SEARCH);
        }

        @Override
        public VolumeRequest build() {
            return new VolumeRequest(this);
        }
    }
}
