package com.crazzyghost.alphavantage.technicalindicator.response.ppo;

import com.crazzyghost.alphavantage.parser.Parser;
import com.crazzyghost.alphavantage.technicalindicator.response.PriceOscillatorResponse;
import com.crazzyghost.alphavantage.technicalindicator.response.SimpleTechnicalIndicatorUnit;

import java.util.List;
import java.util.Map;

public class PPOResponse extends PriceOscillatorResponse {

    private PPOResponse(List<SimpleTechnicalIndicatorUnit> indicatorUnits, MetaData metaData) {
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
        public PPOResponse get(List<SimpleTechnicalIndicatorUnit> indicatorUnits, MetaData metaData) {
            return new PPOResponse(indicatorUnits, metaData);
        }

        @Override
        public PPOResponse get(String error) {
            return new PPOResponse(error);
        }

        @Override
        public String getTechnicalIndicatorKey() {
            return "PPO";
        }
    }
}
