package com.crazzyghost.alphavantage.alphaIntelligence.response;

import java.util.*;
import java.util.function.BiConsumer;

/**
 * Represents the response from the Alpha Vantage Analytics Fixed Window endpoint.
 * <p>
 * Provides access to metadata, return calculations, and all supported advanced analytics data.
 */
@SuppressWarnings("unchecked")
public class AnalyticsFixedWindowResponse {

    // Error message from API, if any. Set only if API response contains an error.
    private String errorMessage;
    // Meta-information about the API response (symbols, date range, etc.).
    private MetaData metaData;
    // All the statistical analytics (parsed from payload).
    private ReturnsCalculations returnsCalculations;

    /**
     * Contains meta-information about the analytics request and response.
     * Maps directly to the "meta_data" object in the JSON response.
     */
    public static class MetaData {
        public String symbols;   // Comma-separated list of symbols included in the request.
        public String minDt;     // Earliest date/time present in the data window.
        public String maxDt;     // Latest date/time present in the data window.
        public String ohlc;      // Whether OHLC (Open/High/Low/Close) data was included ("true"/"false").
        public String interval;  // Data granularity (e.g., "1d", "5min").

        /**
         * Constructs a MetaData object from a map.
         *
         * @param map JSON map for "meta_data" section.
         */
        public MetaData(Map<String, Object> map) {
            this.symbols = (String) map.getOrDefault("symbols", null);
            this.minDt = (String) map.getOrDefault("min_dt", null);
            this.maxDt = (String) map.getOrDefault("max_dt", null);
            this.ohlc = (String) map.getOrDefault("ohlc", null);
            this.interval = (String) map.getOrDefault("interval", null);
        }
    }

    /**
     * Contains all statistical calculations returned by the endpoint,
     * including advanced analytics such as min, max, median, cumulative return, variance,
     * max drawdown, histogram, autocorrelation, covariance, and correlation.
     */
    public static class ReturnsCalculations {
        // Each metric is stored as a map: symbol -> value.

        public Map<String, Double> min = new HashMap<>();                // Minimum values for each symbol.
        public Map<String, Double> max = new HashMap<>();                // Maximum values for each symbol.
        public Map<String, Double> mean = new HashMap<>();               // Mean values for each symbol.
        public Map<String, Double> median = new HashMap<>();             // Median values for each symbol.
        public Map<String, Double> cumulativeReturn = new HashMap<>();   // Cumulative return for each symbol.
        public Map<String, Double> variance = new HashMap<>();           // Variance for each symbol.
        public Map<String, Double> stddev = new HashMap<>();             // Standard deviation for each symbol.
        public Map<String, Double> maxDrawdown = new HashMap<>();        // Maximum drawdown for each symbol.

        // Advanced analytics per symbol.
        public Map<String, HistogramResult> histogram = new HashMap<>(); // Histogram bins/frequencies for each symbol.
        public Map<String, Map<String, Double>> autocorrelation = new HashMap<>(); // Lag -> autocorrelation value per symbol.

        // Matrix analytics for multi-symbol cases (correlation/covariance).
        public MatrixResult covariance;      // Covariance matrix between all symbols.
        public Correlation correlation;      // Correlation matrix between all symbols.

        /**
         * Constructs ReturnsCalculations by parsing a JSON-like map structure.
         *
         * @param map the "RETURNS_CALCULATIONS" section of the response JSON.
         */
        public ReturnsCalculations(Map<String, Object> map) {
            // Helper: parse a simple map of symbol -> double for a given metric.
            BiConsumer<String, Map<String, Double>> parseMetric = (key, target) -> {
                if (map.containsKey(key)) {
                    Map<String, Object> src = (Map<String, Object>) map.get(key);
                    for (Map.Entry<String, Object> entry : src.entrySet()) {
                        // Only numbers are accepted, null otherwise.
                        target.put(entry.getKey(), entry.getValue() instanceof Number ? ((Number) entry.getValue()).doubleValue() : null);
                    }
                }
            };

            // Parse all simple per-symbol double metrics:
            parseMetric.accept("MIN", min);
            parseMetric.accept("MAX", max);
            parseMetric.accept("MEAN", mean);
            parseMetric.accept("MEDIAN", median);
            parseMetric.accept("CUMULATIVE_RETURN", cumulativeReturn);
            parseMetric.accept("VARIANCE", variance);
            parseMetric.accept("STDDEV", stddev);
            parseMetric.accept("MAX_DRAWDOWN", maxDrawdown);

            // Parse HISTOGRAM: symbol -> { bins: [...], frequencies: [...] }
            if (map.containsKey("HISTOGRAM")) {
                Map<String, Object> histMap = (Map<String, Object>) map.get("HISTOGRAM");
                for (Map.Entry<String, Object> entry : histMap.entrySet()) {
                    histogram.put(entry.getKey(), new HistogramResult((Map<String, Object>) entry.getValue()));
                }
            }

            // Parse AUTOCORRELATION: symbol -> { lag1: value, lag2: value, ... }
            if (map.containsKey("AUTOCORRELATION")) {
                Map<String, Object> acMap = (Map<String, Object>) map.get("AUTOCORRELATION");
                for (Map.Entry<String, Object> entry : acMap.entrySet()) {
                    Map<String, Double> lagMap = new HashMap<>();
                    Map<String, Object> rawLagMap = (Map<String, Object>) entry.getValue();
                    for (Map.Entry<String, Object> lagEntry : rawLagMap.entrySet()) {
                        lagMap.put(lagEntry.getKey(),
                                lagEntry.getValue() instanceof Number ? ((Number) lagEntry.getValue()).doubleValue() : null);
                    }
                    autocorrelation.put(entry.getKey(), lagMap);
                }
            }

            // Parse COVARIANCE matrix, if present.
            if (map.containsKey("COVARIANCE")) {
                covariance = new MatrixResult((Map<String, Object>) map.get("COVARIANCE"));
            }
            // Parse CORRELATION matrix, if present.
            if (map.containsKey("CORRELATION")) {
                correlation = new Correlation((Map<String, Object>) map.get("CORRELATION"));
            }
        }
    }

    /**
     * Represents a histogram result for a symbol: bins and frequencies.
     * <p>
     * Each histogram result consists of:
     * - bins: List of bin edges (as doubles), representing the numeric boundaries for each bin.
     * - frequencies: List of frequencies (as integers), indicating the count of data points in each bin.
     */
    public static class HistogramResult {
        public List<Double> bins;        // Numeric boundaries for histogram bins.
        public List<Integer> frequencies;// Count of data points for each bin.

        /**
         * Constructs a HistogramResult from a map (parsed from JSON).
         * Expects keys "bins" and "frequencies" each mapping to a list.
         */
        public HistogramResult(Map<String, Object> map) {
            bins = new ArrayList<>();
            frequencies = new ArrayList<>();
            // Parse "bins" as List<Double>
            if (map.containsKey("bins")) {
                for (Object d : (List<?>) map.get("bins")) {
                    bins.add(d instanceof Number ? ((Number) d).doubleValue() : null);
                }
            }
            // Parse "frequencies" as List<Integer>
            if (map.containsKey("frequencies")) {
                for (Object i : (List<?>) map.get("frequencies")) {
                    frequencies.add(i instanceof Number ? ((Number) i).intValue() : null);
                }
            }
        }
    }

    /**
     * Represents a generic matrix result, such as covariance matrices.
     * <p>
     * - index: List of strings (e.g., symbol names or variable names), indicating row/column order.
     * - matrix: List of rows, each a List of Double, forming a square or rectangular matrix.
     * (matrix.get(i).get(j) = value for index[i] vs index[j])
     */
    public static class MatrixResult {
        public List<String> index;         // Names of rows/columns in matrix (e.g., symbol list)
        public List<List<Double>> matrix;  // The actual matrix data (row-major)

        /**
         * Constructs a MatrixResult from a map with "index" and "matrix" keys.
         */
        public MatrixResult(Map<String, Object> map) {
            // Parse the index (default to empty list if missing)
            index = (List<String>) map.getOrDefault("index", Collections.emptyList());
            matrix = new ArrayList<>();
            // Parse the matrix (list of lists of numbers)
            List<Object> raw = (List<Object>) map.getOrDefault("matrix", Collections.emptyList());
            for (Object row : raw) {
                List<Double> parsedRow = new ArrayList<>();
                if (row instanceof List) {
                    for (Object d : (List<?>) row) {
                        parsedRow.add(d instanceof Number ? ((Number) d).doubleValue() : null);
                    }
                }
                matrix.add(parsedRow);
            }
        }
    }

    /**
     * Represents a correlation matrix between different symbols' returns.
     * - index: List of symbols in the same order as correlationMatrix rows/columns.
     * - correlationMatrix: correlationMatrix[i][j] = correlation between index[i] and index[j].
     */
    public static class Correlation {
        public List<String> index;                 // Symbol order for the matrix
        public List<List<Double>> correlationMatrix; // The correlation values, row-major

        /**
         * Constructs a Correlation object from a map with "index" and "correlation" keys.
         */
        public Correlation(Map<String, Object> map) {
            index = (List<String>) map.getOrDefault("index", Collections.emptyList());
            correlationMatrix = new ArrayList<>();
            // Parse the "correlation" key (matrix data)
            List<Object> raw = (List<Object>) map.getOrDefault("correlation", Collections.emptyList());
            for (Object row : raw) {
                List<Double> parsedRow = new ArrayList<>();
                if (row instanceof List) {
                    for (Object d : (List<?>) row) {
                        parsedRow.add(d instanceof Number ? ((Number) d).doubleValue() : null);
                    }
                }
                correlationMatrix.add(parsedRow);
            }
        }
    }

    // --- Core construction and API ---

    /**
     * Default constructor for empty response.
     * (Populated via the of() static factory method.)
     */
    public AnalyticsFixedWindowResponse() {
    }

    /**
     * Static factory method to construct a response from a parsed JSON map.
     * - Checks for error messages.
     * - Parses "meta_data" and "payload/RETURNS_CALCULATIONS" if present.
     * - Returns a fully populated AnalyticsFixedWindowResponse object.
     *
     * @param data The root map parsed from JSON response.
     * @return An instance of AnalyticsFixedWindowResponse.
     */
    public static AnalyticsFixedWindowResponse of(Map<String, Object> data) {
        AnalyticsFixedWindowResponse response = new AnalyticsFixedWindowResponse();
        // If API returned an error, set error message and return early.
        if (data.containsKey("Error Message")) {
            response.errorMessage = (String) data.get("Error Message");
            return response;
        }
        // Parse meta data (if present)
        if (data.containsKey("meta_data")) {
            response.metaData = new MetaData((Map<String, Object>) data.get("meta_data"));
        }
        // Parse the payload and its advanced analytics (if present)
        if (data.containsKey("payload")) {
            Map<String, Object> payload = (Map<String, Object>) data.get("payload");
            if (payload.containsKey("RETURNS_CALCULATIONS")) {
                response.returnsCalculations = new ReturnsCalculations(
                        (Map<String, Object>) payload.get("RETURNS_CALCULATIONS")
                );
            }
        }
        return response;
    }

    // --- Getters for client code (read-only API) ---

    public String getErrorMessage() {
        return errorMessage;
    }

    public MetaData getMetaData() {
        return metaData;
    }

    public ReturnsCalculations getReturnsCalculations() {
        return returnsCalculations;
    }
}