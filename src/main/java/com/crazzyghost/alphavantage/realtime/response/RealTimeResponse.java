package com.crazzyghost.alphavantage.realtime.response;

import com.crazzyghost.alphavantage.parser.Parser;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Represents the response object for Alpha Vantage's real-time bulk quote API.
 * <p>
 * Contains the parsed results (list of {@link RealTimeMatch}) and any error messages
 * that may have occurred during parsing. This class provides a static factory method
 * to create itself from a raw response map, and includes a nested parser class for internal use.
 */
public class RealTimeResponse {
    /**
     * Holds an error message if parsing failed or API returned an error; otherwise null.
     */
    private final String errorMessage;

    /**
     * List of successfully parsed real-time quote matches.
     */
    private final List<RealTimeMatch> matches;

    /**
     * Constructs a RealTimeResponse object with given matches and error message.
     *
     * @param matches List of RealTimeMatch, may be null or empty if parsing fails.
     * @param error   Error message string, null if no error occurred.
     */
    private RealTimeResponse(List<RealTimeMatch> matches, String error) {
        this.errorMessage = error;
        this.matches = matches != null ? matches : new ArrayList<>();
    }

    /**
     * Static factory method to create a RealTimeResponse from a map
     * (usually a raw JSON response from the Alpha Vantage API, already deserialized).
     * <p>
     * Internally delegates to {@link RealTimeParser}.
     *
     * @param stringObjectMap Map containing the API response.
     * @return Parsed RealTimeResponse object, with matches and possible error.
     */
    public static RealTimeResponse of(Map<String, Object> stringObjectMap) {
        Parser<RealTimeResponse> parser = new RealTimeParser();
        return parser.parse(stringObjectMap);
    }

    /**
     * Returns the list of real-time matches parsed from the API response.
     * May be empty if parsing failed or API returned no data.
     *
     * @return List of {@link RealTimeMatch} objects.
     */
    public List<RealTimeMatch> getMatches() {
        return matches;
    }

    /**
     * Returns the error message if parsing or response failed.
     * Returns null if there was no error.
     *
     * @return Error message string, or null if no error occurred.
     */
    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Returns a string representation of the RealTimeResponse, using the matches' toString.
     *
     * @return String containing all matches' information.
     */
    @Override
    public String toString() {
        return matches.toString(); // Utilize the overridden toString in RealTimeMatch
    }

    /**
     * Represents a single real-time quote entry (i.e., data for one symbol).
     * Immutable data object with all main price and volume attributes.
     */
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

        /**
         * Constructs a new RealTimeMatch object with all data fields.
         *
         * @param symbol                     Symbol for the security (e.g., "AAPL").
         * @param timestamp                  Timestamp of the quote.
         * @param open                       Opening price for the session.
         * @param high                       Highest price during the session.
         * @param low                        Lowest price during the session.
         * @param close                      Closing price for the session.
         * @param previousClose              Previous session's closing price.
         * @param change                     Change in price (absolute).
         * @param changePercent              Change in price (percentage).
         * @param extendedHoursQuote         Quote during extended hours (if available).
         * @param extendedHoursChange        Change during extended hours.
         * @param extendedHoursChangePercent Percent change during extended hours.
         * @param volume                     Trading volume.
         */
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

        /**
         * @return Symbol for this quote (e.g., "AAPL")
         */
        public String getSymbol() {
            return symbol;
        }

        /**
         * @return Timestamp for this quote, in ISO 8601 format (e.g., "2024-05-16 17:45:00")
         */
        public String getTimestamp() {
            return timestamp;
        }

        /**
         * @return Opening price
         */
        public double getOpen() {
            return open;
        }

        /**
         * @return Highest price for the session
         */
        public double getHigh() {
            return high;
        }

        /**
         * @return Lowest price for the session
         */
        public double getLow() {
            return low;
        }

        /**
         * @return Closing price
         */
        public double getClose() {
            return close;
        }

        /**
         * @return Previous session's closing price
         */
        public double getPreviousClose() {
            return previousClose;
        }

        /**
         * @return Change in price from previous close (absolute)
         */
        public double getChange() {
            return change;
        }

        /**
         * @return Change in price from previous close (percent)
         */
        public double getChangePercent() {
            return changePercent;
        }

        /**
         * @return Extended hours quote (if available)
         */
        public double getExtendedHoursQuote() {
            return extendedHoursQuote;
        }

        /**
         * @return Extended hours price change (absolute)
         */
        public double getExtendedHoursChange() {
            return extendedHoursChange;
        }

        /**
         * @return Extended hours price change (percent)
         */
        public double getExtendedHoursChangePercent() {
            return extendedHoursChangePercent;
        }

        /**
         * @return Volume of trades for this period
         */
        public double getVolume() {
            return volume;
        }

        /**
         * Returns a string with all fields for easy debugging or logging.
         *
         * @return String representation of this match.
         */
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

    /**
     * Internal parser class used for converting raw API response maps
     * into a RealTimeResponse object.
     */
    public static class RealTimeParser extends Parser<RealTimeResponse> {

        /**
         * Parses the given raw response map (typically deserialized from JSON) into a RealTimeResponse object.
         * Handles all necessary type conversions and error handling.
         *
         * @param stringObjectMap Raw response map from the API
         * @return RealTimeResponse containing parsed data and/or error message
         */
        @Override
        @SuppressWarnings("unchecked")
        public RealTimeResponse parse(Map<String, Object> stringObjectMap) {
            List<RealTimeMatch> matches = new ArrayList<>();
            String errorMessage = null;

            if (stringObjectMap.isEmpty()) {
                return onParseError("Empty JSON returned by the API, no data found.");
            }

            try {
                // "data" is expected to be a List of Maps (String, String)
                List<Map<String, String>> quotes = (List<Map<String, String>>) stringObjectMap.get("data");
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

        /**
         * Helper method to safely parse a string as a double.
         * Returns 0.0 if the string is null or parsing fails.
         *
         * @param value String value to parse
         * @return Parsed double value, or 0.0 if null/invalid
         */
        private double parseDouble(String value) {
            try {
                return value != null ? Double.parseDouble(value) : 0.0;
            } catch (NumberFormatException e) {
                return 0.0; // default value if parsing fails
            }
        }

        /**
         * Creates a RealTimeResponse indicating a parsing error, with a given error message.
         *
         * @param error Error message string
         * @return RealTimeResponse containing no matches and the error message
         */
        @Override
        public RealTimeResponse onParseError(String error) {
            return new RealTimeResponse(null, error);
        }
    }
}