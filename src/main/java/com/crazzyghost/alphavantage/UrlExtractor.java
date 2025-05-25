/*
 *
 * Copyright (c) 2020 Sylvester Sefa-Yeboah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package com.crazzyghost.alphavantage;

import okhttp3.Request;

import java.lang.reflect.Field;


/**
 * Utility class to extract and construct valid Alpha Vantage API URLs
 * from request objects. Uses Java reflection to dynamically access fields
 * in the request classes and builds a query string using their values.
 *
 * <p>
 * This class also contains special handling for the Advanced Analytics endpoints
 * (e.g., ANALYTICS_FIXED_WINDOW), which require case-sensitive (UPPERCASE) parameter names
 * as per the Alpha Vantage API documentation. All other endpoints default to lowercase
 * field names in query parameters.
 * </p>
 *
 * <p>
 * Usage Example:
 * <pre>
 *     MyRequest req = new MyRequest(...);
 *     String url = UrlExtractor.extract(req) + myApiKey;
 * </pre>
 * </p>
 *
 * @author Sylvester Sefa-Yeboah & Crecker
 * @since 1.0.0
 */
public class UrlExtractor {

    /**
     * Private constructor to prevent instantiation.
     */
    private UrlExtractor() {
    }

    /**
     * Generates a URL query string from a request object by extracting
     * its fields and formatting them as API parameters.
     * <p>
     * For advanced analytics requests (AnalyticsFixedWindowRequest), specific field
     * names are converted to UPPERCASE to meet Alpha Vantage's requirements.
     * All other fields are rendered in lowercase.
     * </p>
     *
     * @param object A request object with the valid API parameters as fields
     * @return A URL query string (without the base URL) ending with "apikey="
     */
    public static String extract(Object object) {
        // Holds the generated query string (excluding base URL and api key)
        StringBuilder stringBuilder = new StringBuilder();

        // Check if the request is for an analytics endpoint that needs uppercase fields
        boolean isAnalytics = object instanceof com.crazzyghost.alphavantage.alphaIntelligence.request.AnalyticsFixedWindowRequest;

        // Traverse all classes in the request's hierarchy (handles inheritance)
        Class<?> cls = object.getClass();
        while (cls != null) {
            // Access all declared fields in this class (including private fields)
            Field[] fields = cls.getDeclaredFields();
            for (Field field : fields) {
                field.setAccessible(true); // Make private fields accessible
                try {
                    // Only include non-synthetic, non-null fields
                    if (!field.isSynthetic() && field.get(object) != null) {

                        String fieldName = field.getName();
                        // For analytics requests, certain field names are UPPERCASE in the API
                        if (isAnalytics && (
                                fieldName.equals("symbols") ||
                                        fieldName.equals("range") ||
                                        fieldName.equals("interval") ||
                                        fieldName.equals("calculations") ||
                                        fieldName.equals("ohlc")
                        )) {
                            stringBuilder.append(fieldName.toUpperCase()).append("=");
                        }
                        // Always use "function" in lowercase for API calls
                        else if (fieldName.equals("function")) {
                            stringBuilder.append("function").append("=");
                        }
                        // All other parameters are lowercase in the API
                        else {
                            stringBuilder.append(fieldName.toLowerCase()).append("=");
                        }
                        // Append the value of the field
                        String value = (field.get(object)).toString();
                        stringBuilder.append(value).append("&");
                    }
                } catch (IllegalAccessException e) {
                    throw new AlphaVantageException(e.getLocalizedMessage());
                }
            }
            // Move to superclass (to handle inherited fields)
            cls = cls.getSuperclass();
        }
        // End with apikey parameter, but without the actual value
        return stringBuilder.append("apikey=").toString();
    }

    /**
     * Constructs a full HTTP {@link Request} for Alpha Vantage API using the given
     * request object and API key.
     * <p>
     * The generated URL is printed to standard output for debugging purposes.
     * </p>
     *
     * @param request The request object containing API parameters
     * @param apiKey  The Alpha Vantage API key (appended to the URL)
     * @return An OkHttp {@link Request} object ready to be executed
     */
    public static Request extract(Object request, String apiKey) {
        // Compose the full URL using the base URL, query string, and API key
        String url = Config.BASE_URL + UrlExtractor.extract(request) + apiKey;
        // Print the generated URL for debugging
        System.out.println(url);
        // Return a new OkHttp Request object using the full URL
        return new Request.Builder().url(url).build();
    }
}