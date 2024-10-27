package com.crazzyghost.alphavantage.stock.response;

import com.crazzyghost.alphavantage.parser.Parser;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class StockResponse {
    private final String errorMessage;
    private final List<StockMatch> matches; // Change to hold multiple matches

    private StockResponse(List<StockMatch> matches, String error) {
        this.errorMessage = error;
        this.matches = matches != null ? matches : new ArrayList<>();
    }

    public static StockResponse of(Map<String, Object> stringObjectMap) {
        Parser<StockResponse> parser = new StockParser();
        return parser.parse(stringObjectMap);
    }

    public List<StockMatch> getMatches() {
        return matches;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    @Override
    public String toString() {
        return matches.toString(); // Utilize the overridden toString in StockMatch
    }

    public static class StockMatch {
        private final String symbol;
        private final String name;

        public StockMatch(String symbol, String name) {
            this.symbol = symbol;
            this.name = name;
        }

        public String getSymbol() {
            return symbol;
        }

        public String getName() {
            return name;
        }

        @Override
        public String toString() {
            return symbol + " - " + name; // String representation of the match
        }
    }

    public static class StockParser extends Parser<StockResponse> {

        @SuppressWarnings("unchecked")
        @Override
        public StockResponse parse(Map<String, Object> stringObjectMap) {
            List<StockMatch> matches = new ArrayList<>();
            String errorMessage = null;

            // Check if the response is empty
            if (stringObjectMap.isEmpty()) {
                return onParseError("Empty JSON returned by the API, no matches found.");
            }

            // Attempt to parse the bestMatches
            try {
                List<Map<String, String>> bestMatches = (List<Map<String, String>>) stringObjectMap.get("bestMatches");
                if (bestMatches != null) {
                    for (Map<String, String> match : bestMatches) {
                        matches.add(new StockMatch(
                                match.get("1. symbol"),
                                match.get("2. name")
                        ));
                    }
                }
            } catch (ClassCastException | NullPointerException e) {
                errorMessage = "Error parsing symbol search results.";
            }

            // Return new StockResponse with matches and error message if any
            return new StockResponse(matches, errorMessage);
        }

        @Override
        public StockResponse onParseError(String error) {
            return new StockResponse(null, error);
        }
    }
}