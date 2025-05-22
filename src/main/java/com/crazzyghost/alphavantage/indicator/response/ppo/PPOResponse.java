package com.crazzyghost.alphavantage.indicator.response.ppo;

import com.crazzyghost.alphavantage.indicator.response.PriceOscillatorResponse;
import com.crazzyghost.alphavantage.indicator.response.SimpleIndicatorUnit;
import com.crazzyghost.alphavantage.parser.Parser;

import java.util.List;
import java.util.Map;

public class PPOResponse extends PriceOscillatorResponse {

    private PPOResponse(List<SimpleIndicatorUnit> indicatorUnits, MetaData metaData) {
        super(indicatorUnits, metaData);
    }

    private PPOResponse(String errorMessage) {
        super(errorMessage);
    }

    public static PPOResponse of(Map<String, Object> stringObjectMap) {
        Parser<PPOResponse> parser = new PPOParser();
        return parser.parse(stringObjectMap);
    }

    public static class PPOParser extends PriceOscillatorParser<PPOResponse> {
        @Override
        public PPOResponse get(List<SimpleIndicatorUnit> indicatorUnits, MetaData metaData) {
            return new PPOResponse(indicatorUnits, metaData);
        }

        @Override
        public PPOResponse get(String error) {
            return new PPOResponse(error);
        }

        @Override
        public String getIndicatorKey() {
            return "PPO";
        }
    }
}
