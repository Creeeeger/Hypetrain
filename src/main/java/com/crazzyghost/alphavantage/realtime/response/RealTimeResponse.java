package com.crazzyghost.alphavantage.realtime.response;

import com.crazzyghost.alphavantage.parser.Parser;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class RealTimeResponse {
    private final String errorMessage;
    private final List<RealTimeMatch> matches;

    private RealTimeResponse(List<RealTimeMatch> matches, String error) {
        this.errorMessage = error;
        this.matches = matches != null ? matches : new ArrayList<>();
    }

    public static RealTimeResponse of(Map<String, Object> stringObjectMap) {
        Parser<RealTimeResponse> parser = new RealTimeParser();
        return parser.parse(stringObjectMap);
    }

    public List<RealTimeMatch> getMatches() {
        return matches;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    @Override
    public String toString() {
        return matches.toString(); // Utilize the overridden toString in RealTimeMatch
    }

    public static class RealTimeMatch {
        private final String symbol;
        private final String timestamp;
        private final double open;
        private final double high;
        private final double low;
        private final double close;
        private final double previousClose;
        private final double change;
        private final double changePercent;
        private final double extendedHoursQuote;
        private final double extendedHoursChange;
        private final double extendedHoursChangePercent;
        private final double volume;

        public RealTimeMatch(String symbol, String timestamp, double open, double high, double low, double close,
                             double previousClose, double change, double changePercent, double extendedHoursQuote,
                             double extendedHoursChange, double extendedHoursChangePercent, double volume) {
            this.symbol = symbol;
            this.timestamp = timestamp;
            this.open = open;
            this.high = high;
            this.low = low;
            this.close = close;
            this.previousClose = previousClose;
            this.change = change;
            this.changePercent = changePercent;
            this.extendedHoursQuote = extendedHoursQuote;
            this.extendedHoursChange = extendedHoursChange;
            this.extendedHoursChangePercent = extendedHoursChangePercent;
            this.volume = volume;
        }

        public String getSymbol() {
            return symbol;
        }

        public String getTimestamp() {
            return timestamp;
        }

        public double getOpen() {
            return open;
        }

        public double getHigh() {
            return high;
        }

        public double getLow() {
            return low;
        }

        public double getClose() {
            return close;
        }

        public double getPreviousClose() {
            return previousClose;
        }

        public double getChange() {
            return change;
        }

        public double getChangePercent() {
            return changePercent;
        }

        public double getExtendedHoursQuote() {
            return extendedHoursQuote;
        }

        public double getExtendedHoursChange() {
            return extendedHoursChange;
        }

        public double getExtendedHoursChangePercent() {
            return extendedHoursChangePercent;
        }

        public double getVolume() {
            return volume;
        }

        @Override
        public String toString() {
            return "Symbol: " + symbol +
                    ", Timestamp: " + timestamp +
                    ", Open: " + open +
                    ", High: " + high +
                    ", Low: " + low +
                    ", Close: " + close +
                    ", Previous Close: " + previousClose +
                    ", Change: " + change +
                    ", Change Percent: " + changePercent +
                    ", Extended Hours Quote: " + extendedHoursQuote +
                    ", Extended Hours Change: " + extendedHoursChange +
                    ", Extended Hours Change Percent: " + extendedHoursChangePercent +
                    ", Volume: " + volume;
        }
    }

    public static class RealTimeParser extends Parser<RealTimeResponse> {

        @Override
        @SuppressWarnings("unchecked")
        public RealTimeResponse parse(Map<String, Object> stringObjectMap) {
            List<RealTimeMatch> matches = new ArrayList<>();
            String errorMessage = null;

            if (stringObjectMap.isEmpty()) {
                return onParseError("Empty JSON returned by the API, no data found.");
            }

            try {
                List<Map<String, String>> quotes = (List<Map<String, String>>) stringObjectMap.get("Realtime Quotes");
                if (quotes != null) {
                    for (Map<String, String> quote : quotes) {
                        matches.add(new RealTimeMatch(
                                quote.get("symbol"),
                                quote.get("timestamp"),
                                parseDouble(quote.get("open")),
                                parseDouble(quote.get("high")),
                                parseDouble(quote.get("low")),
                                parseDouble(quote.get("close")),
                                parseDouble(quote.get("previous_close")),
                                parseDouble(quote.get("change")),
                                parseDouble(quote.get("change_percent")),
                                parseDouble(quote.get("extended_hours_quote")),
                                parseDouble(quote.get("extended_hours_change")),
                                parseDouble(quote.get("extended_hours_change_percent")),
                                parseDouble(quote.get("volume"))
                        ));
                    }
                }
            } catch (ClassCastException | NullPointerException | NumberFormatException e) {
                errorMessage = "Error parsing real-time quotes.";
            }

            return new RealTimeResponse(matches, errorMessage);
        }

        // Helper method for parsing strings to double with null handling
        private double parseDouble(String value) {
            try {
                return value != null ? Double.parseDouble(value) : 0.0;
            } catch (NumberFormatException e) {
                return 0.0; // default value if parsing fails
            }
        }

        @Override
        public RealTimeResponse onParseError(String error) {
            return new RealTimeResponse(null, error);
        }
    }
}