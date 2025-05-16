package com.crazzyghost.alphavantage.stock.response;

import com.crazzyghost.alphavantage.parser.Parser;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Represents the response object for Alpha Vantage's stock symbol search API.
 * <p>
 * Contains a list of {@link StockMatch} objects representing matching stocks,
 * along with an optional error message if parsing failed or no data was found.
 * Instances are typically constructed by calling {@link #of(Map)} with a parsed JSON response.
 */
public class StockResponse {
    /**
     * Holds an error message if parsing fails, or null if successful.
     */
    private final String errorMessage;

    /**
     * List of parsed stock matches from the API response.
     */
    private final List<StockMatch> matches;

    /**
     * Constructs a StockResponse with given matches and error message.
     *
     * @param matches List of {@link StockMatch} or null.
     * @param error   Error message string or null if no error occurred.
     */
    private StockResponse(List<StockMatch> matches, String error) {
        this.errorMessage = error;
        this.matches = matches != null ? matches : new ArrayList<>();
    }

    /**
     * Factory method to create a StockResponse from a raw response map.
     * <p>
     * Internally delegates to {@link StockParser} to parse the data.
     *
     * @param stringObjectMap Map (usually from deserialized JSON) containing the API response.
     * @return Parsed {@link StockResponse} object.
     */
    public static StockResponse of(Map<String, Object> stringObjectMap) {
        Parser<StockResponse> parser = new StockParser();
        return parser.parse(stringObjectMap);
    }

    /**
     * Returns the list of matching stocks found in the response.
     *
     * @return List of {@link StockMatch}. Never null, but may be empty.
     */
    public List<StockMatch> getMatches() {
        return matches;
    }

    /**
     * Returns an error message if the API response failed or parsing failed.
     *
     * @return Error message string, or null if there was no error.
     */
    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Returns a string representation of the matches list for easy viewing or debugging.
     *
     * @return String showing all stock matches.
     */
    @Override
    public String toString() {
        return matches.toString(); // Utilizes StockMatch.toString()
    }

    /**
     * Represents a single stock match result from the Alpha Vantage API.
     * Immutable data holder for a stock's symbol and name.
     */
    public static class StockMatch {
        /**
         * The stock's trading symbol (e.g., "AAPL").
         */
        private final String symbol;
        /**
         * The full name of the company or stock (e.g., "Apple Inc.").
         */
        private final String name;

        /**
         * Constructs a StockMatch for the given symbol and name.
         *
         * @param symbol Trading symbol (e.g., "AAPL").
         * @param name   Full stock/company name.
         */
        public StockMatch(String symbol, String name) {
            this.symbol = symbol;
            this.name = name;
        }

        /**
         * @return The trading symbol of this match.
         */
        public String getSymbol() {
            return symbol;
        }

        /**
         * @return The full name of the matched company/stock.
         */
        public String getName() {
            return name;
        }

        /**
         * Returns a short string representation, for example: "AAPL - Apple Inc."
         *
         * @return Symbol and name combined.
         */
        @Override
        public String toString() {
            return symbol + " - " + name;
        }
    }

    /**
     * Internal parser class used to convert a raw API map into a {@link StockResponse}.
     */
    public static class StockParser extends Parser<StockResponse> {

        /**
         * Parses a raw response map (from JSON) into a {@link StockResponse}.
         * <p>
         * Expects a "bestMatches" field in the root map, containing a list of matches.
         * Each match should be a map with at least "1. symbol" and "2. name" fields.
         *
         * @param stringObjectMap Raw map from the API response.
         * @return Parsed StockResponse with matches and error (if any).
         */
        @SuppressWarnings("unchecked")
        @Override
        public StockResponse parse(Map<String, Object> stringObjectMap) {
            List<StockMatch> matches = new ArrayList<>();
            String errorMessage = null;

            // Check if the response is empty
            if (stringObjectMap.isEmpty()) {
                return onParseError("Empty JSON returned by the API, no matches found.");
            }

            // Attempt to parse the "bestMatches" list from the response
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

            // Return new StockResponse with matches and error message (if any)
            return new StockResponse(matches, errorMessage);
        }

        /**
         * Creates a StockResponse representing a parsing error.
         *
         * @param error Error message string.
         * @return {@link StockResponse} with no matches and the given error.
         */
        @Override
        public StockResponse onParseError(String error) {
            return new StockResponse(null, error);
        }
    }
}