package com.crazzyghost.alphavantage.alphaIntelligence.response;

import java.util.*;

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
     * Contains all statistical calculations and advanced analytics returned by
     * the Alpha Vantage Analytics Fixed Window endpoint.
     * <p>
     * This class is designed to be future-proof:
     * <ul>
     *   <li>It dynamically parses all present and future per-symbol metrics from the API response,
     *       so adding new calculations does not require code changes.</li>
     *   <li>It provides explicit fields for well-known metrics (mean, stddev, autocorrelation, etc.) for convenience.</li>
     *   <li>All unknown or custom metrics are available via generic dynamic maps.</li>
     *   <li>Special handling is included for matrix-valued and parameterized metrics (e.g., correlation, autocorrelation, histogram).</li>
     * </ul>
     */
    public static class ReturnsCalculations {

        /**
         * Dynamic map: metric name → { symbol → value }, e.g. "MEAN" → { "AAPL" → 0.01, ... }
         */
        public Map<String, Map<String, Double>> perSymbolMetrics = new HashMap<>();

        /**
         * Dynamic map for all unknown or new metrics (for future proofing).
         */
        public Map<String, Map<String, Object>> genericMetrics = new HashMap<>();

        // ======= Explicit fields for common, well-known metrics: =======

        /**
         * Map from symbol to MIN value.
         */
        public Map<String, Double> min = new HashMap<>();

        /**
         * Map from symbol to MAX value.
         */
        public Map<String, Double> max = new HashMap<>();

        /**
         * Map from symbol to MEAN value.
         */
        public Map<String, Double> mean = new HashMap<>();

        /**
         * Map from symbol to MEDIAN value.
         */
        public Map<String, Double> median = new HashMap<>();

        /**
         * Map from symbol to CUMULATIVE_RETURN value.
         */
        public Map<String, Double> cumulativeReturn = new HashMap<>();

        /**
         * Map from symbol to VARIANCE value.
         */
        public Map<String, Double> variance = new HashMap<>();

        /**
         * Map from symbol to STDDEV value.
         */
        public Map<String, Double> stddev = new HashMap<>();

        /**
         * Map from symbol to MAX_DRAWDOWN value.
         */
        public Map<String, Double> maxDrawdown = new HashMap<>();

        // ======= Advanced / parameterized metrics =======

        /**
         * Map from symbol to metric name (with params) to value, e.g.
         * "AAPL" → { "AUTOCORRELATION(LAG=1)": 0.8, "AUTOCORRELATION(LAG=2)": 0.7 }
         * Use this for any parameterized or special calculation.
         */
        public Map<String, Map<String, Double>> autocorrelation = new HashMap<>();

        /**
         * Map from symbol to histogram result (contains bins and frequencies).
         * For HISTOGRAM and HISTOGRAM(bins=...)
         */
        public Map<String, HistogramResult> histogram = new HashMap<>();

        /**
         * Covariance matrix between all requested symbols, if present.
         */
        public MatrixResult covariance;

        /**
         * Correlation matrix between all requested symbols, if present.
         */
        public Correlation correlation;

        /**
         * Constructs the ReturnsCalculations object by parsing all metrics from the
         * "RETURNS_CALCULATIONS" section of the Alpha Vantage API response.
         *
         * @param map The JSON-like map containing all metric entries as returned by the API.
         */
        public ReturnsCalculations(Map<String, Object> map) {
            // Iterate over all top-level keys in the "RETURNS_CALCULATIONS" map.
            for (Map.Entry<String, Object> entry : map.entrySet()) {
                String key = entry.getKey();
                Object val = entry.getValue();

                // --- 1. Handle all per-symbol double metrics: e.g., "MEAN", "STDDEV", etc. ---
                if (isSimpleMetric(key)) {
                    // The value is a map: symbol → value (number)
                    Map<String, Object> rawMap = castToMap(val);
                    Map<String, Double> metricMap = new HashMap<>();
                    for (Map.Entry<String, Object> e : rawMap.entrySet()) {
                        metricMap.put(e.getKey(), e.getValue() instanceof Number ? ((Number) e.getValue()).doubleValue() : null);
                    }
                    // Save in the dynamic map for general access
                    perSymbolMetrics.put(key, metricMap);

                    // Also keep explicit references for common metrics (legacy/convenience)
                    switch (key) {
                        case "MIN":
                            min = metricMap;
                            break;
                        case "MAX":
                            max = metricMap;
                            break;
                        case "MEAN":
                            mean = metricMap;
                            break;
                        case "MEDIAN":
                            median = metricMap;
                            break;
                        case "CUMULATIVE_RETURN":
                            cumulativeReturn = metricMap;
                            break;
                        case "VARIANCE":
                            variance = metricMap;
                            break;
                        case "STDDEV":
                            stddev = metricMap;
                            break;
                        case "MAX_DRAWDOWN":
                            maxDrawdown = metricMap;
                            break;
                    }
                    continue;
                }

                // --- 2. Parameterized metrics: AUTOCORRELATION(LAG=...), etc. ---
                if (key.startsWith("AUTOCORRELATION(")) {
                    Map<String, Object> symbolMap = castToMap(val);
                    for (Map.Entry<String, Object> symEntry : symbolMap.entrySet()) {
                        String symbol = symEntry.getKey();
                        Double value = symEntry.getValue() instanceof Number ? ((Number) symEntry.getValue()).doubleValue() : null;
                        autocorrelation.computeIfAbsent(symbol, k -> new HashMap<>()).put(key, value);
                    }
                    continue;
                }

                // --- 3. HISTOGRAM or HISTOGRAM(bins=...) ---
                if (key.startsWith("HISTOGRAM")) {
                    Map<String, Object> histMap = castToMap(val);
                    for (Map.Entry<String, Object> symEntry : histMap.entrySet()) {
                        histogram.put(symEntry.getKey(), new HistogramResult(castToMap(symEntry.getValue())));
                    }
                    continue;
                }

                // --- 4. Matrix metrics: CORRELATION, COVARIANCE ---
                if (key.equals("CORRELATION")) {
                    correlation = new Correlation(castToMap(val));
                    continue;
                }
                if (key.equals("COVARIANCE")) {
                    covariance = new MatrixResult(castToMap(val));
                    continue;
                }

                // --- 5. Catch-all for any unknown/future metrics ---
                // This ensures future expansion (metrics not explicitly handled above will still be available)
                if (val instanceof Map) {
                    genericMetrics.put(key, castToMap(val));
                }
            }
        }

        /**
         * Checks if the metric is one of the well-known simple per-symbol metrics.
         *
         * @param key The metric name.
         * @return True if the metric is a known per-symbol value.
         */
        private boolean isSimpleMetric(String key) {
            return switch (key) {
                case "MIN", "MAX", "MEAN", "MEDIAN", "CUMULATIVE_RETURN", "VARIANCE", "STDDEV", "MAX_DRAWDOWN" -> true;
                default -> false;
            };
        }

        /**
         * Safely casts an object to a Map<String, Object> (suppresses unchecked warning).
         */
        @SuppressWarnings("unchecked")
        private Map<String, Object> castToMap(Object obj) {
            return (Map<String, Object>) obj;
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