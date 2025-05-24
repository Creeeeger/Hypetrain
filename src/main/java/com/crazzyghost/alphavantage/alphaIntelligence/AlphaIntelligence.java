package com.crazzyghost.alphavantage.alphaIntelligence;

import com.crazzyghost.alphavantage.AlphaVantageException;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.Fetcher;
import com.crazzyghost.alphavantage.UrlExtractor;
import com.crazzyghost.alphavantage.alphaIntelligence.request.AlphaIntelligenceRequest;
import com.crazzyghost.alphavantage.alphaIntelligence.request.AnalyticsFixedWindowRequest;
import com.crazzyghost.alphavantage.alphaIntelligence.request.InsiderTransactionsRequest;
import com.crazzyghost.alphavantage.alphaIntelligence.response.AnalyticsFixedWindowResponse;
import com.crazzyghost.alphavantage.alphaIntelligence.response.InsiderTransactionsResponse;
import com.crazzyghost.alphavantage.parser.Parser;
import okhttp3.Call;
import okhttp3.Response;
import okhttp3.ResponseBody;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;
import java.util.Objects;

/**
 * Client for the Alpha Vantage Alpha Intelligence endpoints.
 *
 * <p>
 * Provides methods to build and execute analytics and insider transactions requests,
 * supporting both asynchronous and synchronous (blocking) fetch patterns.
 *
 * <h2>Usage Example:</h2>
 * <pre>
 * AlphaIntelligence ai = new AlphaIntelligence(config);
 * ai.analyticsFixedWindow()
 *   .symbols("AAPL,MSFT")
 *   .interval("1d")
 *   .calculations("mean,median")
 *   .onSuccess(response -> { ... })
 *   .onFailure(exception -> { ... })
 *   .fetch();
 * </pre>
 */
public final class AlphaIntelligence implements Fetcher {

    /**
     * Configuration for API calls, including API key and OkHttpClient instance.
     * Required to perform network requests and authenticate with the Alpha Vantage API.
     */
    private final Config config;

    // -- Analytics endpoint request builder and callbacks --
    // Stores the builder instance for the current analytics request.
    private AlphaIntelligenceRequest.Builder<?> analyticsBuilder;
    // Stores the user-defined callback to handle successful analytics responses.
    private Fetcher.SuccessCallback<AnalyticsFixedWindowResponse> analyticsSuccessCallback;
    // Stores the user-defined callback to handle analytics request failures.
    private Fetcher.FailureCallback analyticsFailureCallback;

    // -- Insider endpoint request builder and callbacks --
    // Stores the builder instance for the current insider transactions request.
    private AlphaIntelligenceRequest.Builder<?> insiderBuilder;
    // Stores the user-defined callback to handle successful insider responses.
    private Fetcher.SuccessCallback<InsiderTransactionsResponse> insiderSuccessCallback;
    // Stores the user-defined callback to handle insider request failures.
    private Fetcher.FailureCallback insiderFailureCallback;

    /**
     * Constructs an AlphaIntelligence API client with the given configuration.
     *
     * @param config Alpha Vantage API configuration object.
     */
    public AlphaIntelligence(Config config) {
        this.config = config;
    }

    /**
     * Begins building an analytics fixed window request.
     * Returns a proxy object for method chaining.
     *
     * @return AnalyticsFixedWindowRequestProxy for fluent configuration and fetch.
     */
    public AnalyticsFixedWindowRequestProxy analyticsFixedWindow() {
        return new AnalyticsFixedWindowRequestProxy();
    }

    /**
     * Begins building an insider transactions request.
     * Returns a proxy object for method chaining.
     *
     * @return InsiderTransactionsRequestProxy for fluent configuration and fetch.
     */
    public InsiderTransactionsRequestProxy insiderTransactions() {
        return new InsiderTransactionsRequestProxy();
    }

    /**
     * Dispatches the currently prepared request (analytics or insider).
     * Determines which type of request has been built and executes the appropriate fetch logic.
     * Only one request type (analytics or insider) should be active at a time.
     */
    @Override
    public void fetch() {
        if (analyticsBuilder != null) {
            fetchAnalytics(); // If an analytics request is ready, send it.
        } else if (insiderBuilder != null) {
            fetchInsider(); // If an insider request is ready, send it.
        }
        // If neither builder is set, this method does nothing.
    }

    /**
     * Internal helper: Executes an asynchronous analytics request and handles the response with user-provided callbacks.
     * <p>
     * Flow:
     * 1. Validates configuration.
     * 2. Constructs the HTTP call using OkHttp.
     * 3. Enqueues the call for asynchronous execution.
     * 4. Handles both HTTP/network failures and API-level errors with callbacks.
     */
    private void fetchAnalytics() {
        // Ensure the API key and configuration are present and valid.
        Config.checkNotNullOrKeyEmpty(config);

        // Build the HTTP call for the analytics request using OkHttp.
        // UrlExtractor constructs the correct endpoint URL and query parameters using the builder and API key.
        config.getOkHttpClient().newCall(
                UrlExtractor.extract(analyticsBuilder.build(), config.getKey())
        ).enqueue(new okhttp3.Callback() {

            /**
             * Called if the HTTP request fails due to network issues or similar.
             * Passes the exception to the user-defined failure callback.
             */
            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                // If a failure callback was provided, invoke it with the exception.
                if (analyticsFailureCallback != null) {
                    analyticsFailureCallback.onFailure(new AlphaVantageException(e.getMessage()));
                }
            }

            /**
             * Called when a response is received from the server (successful or otherwise).
             * Handles:
             *   - Successful HTTP responses: parses the JSON body.
             *   - API-level errors: if Alpha Vantage returns an error message.
             *   - Unsuccessful HTTP responses (e.g., 4xx or 5xx): treated as generic failures.
             */
            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                // Check for successful HTTP status (2xx).
                if (response.isSuccessful()) {
                    // Parse the response body safely using try-with-resources.
                    try (ResponseBody body = response.body()) {
                        // Parse JSON response string into a map and build an AnalyticsFixedWindowResponse object.
                        AnalyticsFixedWindowResponse apiResponse = AnalyticsFixedWindowResponse.of(
                                Parser.parseJSON(Objects.requireNonNull(body).string())
                        );
                        // If the API response contains an error message, treat as failure.
                        if (apiResponse.getErrorMessage() != null && analyticsFailureCallback != null) {
                            analyticsFailureCallback.onFailure(new AlphaVantageException(apiResponse.getErrorMessage()));
                        }
                        // Otherwise, pass the parsed response object to the success callback.
                        if (analyticsSuccessCallback != null) analyticsSuccessCallback.onSuccess(apiResponse);
                    }
                } else {
                    // If HTTP response code is not successful, treat as a generic failure.
                    if (analyticsFailureCallback != null)
                        analyticsFailureCallback.onFailure(new AlphaVantageException("Unsuccessful HTTP response"));
                }
            }
        });
    }

    /**
     * Internal: Executes async insider request, handles callbacks.
     * <p>
     * This method is responsible for sending an asynchronous HTTP request for the "insider transactions" endpoint.
     * It uses OkHttp's async execution, and invokes user-supplied callbacks for success and failure scenarios.
     */
    private void fetchInsider() {
        // Ensure config is valid and API key is present.
        Config.checkNotNullOrKeyEmpty(config);

        // Create and enqueue an HTTP request using OkHttp.
        config.getOkHttpClient().newCall(
                UrlExtractor.extract(insiderBuilder.build(), config.getKey()) // Builds the URL for the insider request.
        ).enqueue(new okhttp3.Callback() {

            /**
             * Called if the HTTP request fails due to network problems.
             * The user-supplied failure callback is invoked with the exception.
             */
            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                if (insiderFailureCallback != null) {
                    insiderFailureCallback.onFailure(new AlphaVantageException(e.getMessage()));
                }
            }

            /**
             * Called when a server response is received (whether HTTP 200 or not).
             * - On success, attempts to parse the response body as JSON.
             * - Handles API-level errors as well as HTTP-level errors.
             * - Invokes the user callbacks accordingly.
             */
            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    try (ResponseBody body = response.body()) {
                        // Parse response JSON and create the strongly-typed response object.
                        InsiderTransactionsResponse apiResponse = InsiderTransactionsResponse.of(
                                Parser.parseJSON(Objects.requireNonNull(body).string())
                        );
                        // If API error, notify user via failure callback.
                        if (apiResponse.getErrorMessage() != null && insiderFailureCallback != null) {
                            insiderFailureCallback.onFailure(new AlphaVantageException(apiResponse.getErrorMessage()));
                        }
                        // If success, provide the response to the success callback.
                        if (insiderSuccessCallback != null) insiderSuccessCallback.onSuccess(apiResponse);
                    }
                } else {
                    // If HTTP was unsuccessful (non-2xx), notify user via failure callback.
                    if (insiderFailureCallback != null)
                        insiderFailureCallback.onFailure(new AlphaVantageException("Unsuccessful HTTP response"));
                }
            }
        });
    }

    /**
     * Internal: Executes synchronous (blocking) analytics request and delivers result to callback.
     *
     * @param callback Success callback which receives the response object if the call succeeds.
     * @throws AlphaVantageException if there is an IOException (network error).
     *                               <p>
     *                               This method blocks until the request completes.
     */
    private void fetchAnalyticsSync(SuccessCallback<AnalyticsFixedWindowResponse> callback) throws AlphaVantageException {
        Config.checkNotNullOrKeyEmpty(config);  // Ensure config and key.
        this.analyticsSuccessCallback = callback;   // Store callback.
        this.analyticsFailureCallback = null;       // Disable async failure callback (sync only).
        okhttp3.OkHttpClient client = config.getOkHttpClient();
        try (Response response = client.newCall(UrlExtractor.extract(analyticsBuilder.build(), config.getKey())).execute()) {
            // Parse response and hand it to the callback.
            AnalyticsFixedWindowResponse apiResponse = AnalyticsFixedWindowResponse.of(
                    Parser.parseJSON(Objects.requireNonNull(response.body()).string())
            );
            this.analyticsSuccessCallback.onSuccess(apiResponse);
        } catch (IOException e) {
            // Wrap network errors as AlphaVantageException.
            throw new AlphaVantageException(e.getMessage());
        }
    }

    /**
     * Internal: Executes synchronous (blocking) insider request and delivers result to callback.
     *
     * @param callback Success callback which receives the response object if the call succeeds.
     * @throws AlphaVantageException if there is an IOException (network error).
     *                               <p>
     *                               This method blocks until the request completes.
     */
    private void fetchInsiderSync(SuccessCallback<InsiderTransactionsResponse> callback) throws AlphaVantageException {
        Config.checkNotNullOrKeyEmpty(config);   // Validate config and key.
        this.insiderSuccessCallback = callback;  // Store callback.
        this.insiderFailureCallback = null;      // Disable async failure callback (sync only).
        okhttp3.OkHttpClient client = config.getOkHttpClient();
        try (Response response = client.newCall(UrlExtractor.extract(insiderBuilder.build(), config.getKey())).execute()) {
            // Parse response and hand it to the callback.
            InsiderTransactionsResponse apiResponse = InsiderTransactionsResponse.of(
                    Parser.parseJSON(Objects.requireNonNull(response.body()).string())
            );
            this.insiderSuccessCallback.onSuccess(apiResponse);
        } catch (IOException e) {
            // Wrap network errors as AlphaVantageException.
            throw new AlphaVantageException(e.getMessage());
        }
    }

    /**
     * Fluent proxy for building and executing Analytics Fixed Window requests.
     * Supports both asynchronous and synchronous usage.
     */
    public class AnalyticsFixedWindowRequestProxy {

        /**
         * The builder instance used to construct the analytics request.
         * Exposes methods for setting each API parameter.
         */
        protected AnalyticsFixedWindowRequest.Builder builder = new AnalyticsFixedWindowRequest.Builder();
        /**
         * The response object returned by a synchronous fetch.
         * (This is set internally and returned by fetchSync.)
         */
        protected AnalyticsFixedWindowResponse syncResponse;

        /**
         * Sets the target symbols for the analytics request.
         *
         * @param symbols Comma-separated list of symbols (e.g., "AAPL,MSFT")
         * @return this proxy for method chaining
         */
        public AnalyticsFixedWindowRequestProxy symbols(String symbols) {
            builder.symbols(symbols);
            return this;
        }

        /**
         * Sets the time range(s) for the analytics request.
         *
         * @param range One or more range specifiers (e.g., "1d", "5d")
         * @return this proxy for method chaining
         */
        public AnalyticsFixedWindowRequestProxy range(String... range) {
            builder.range(range);
            return this;
        }

        /**
         * Sets the data interval (granularity) for the analytics request.
         *
         * @param interval e.g., "1d", "5min"
         * @return this proxy for method chaining
         */
        public AnalyticsFixedWindowRequestProxy interval(String interval) {
            builder.interval(interval);
            return this;
        }

        /**
         * Sets the calculations to perform (e.g., "mean,median").
         *
         * @param calculations Comma-separated calculation list.
         * @return this proxy for method chaining
         */
        public AnalyticsFixedWindowRequestProxy calculations(String calculations) {
            builder.calculations(calculations);
            return this;
        }

        /**
         * Specifies whether to include OHLC data.
         *
         * @param ohlc "true" or "false"
         * @return this proxy for method chaining
         */
        public AnalyticsFixedWindowRequestProxy ohlc(String ohlc) {
            builder.ohlc(ohlc);
            return this;
        }

        /**
         * Sets the success callback for asynchronous fetch.
         *
         * @param callback Success callback (response object passed on completion)
         * @return this proxy for method chaining
         */
        public AnalyticsFixedWindowRequestProxy onSuccess(SuccessCallback<AnalyticsFixedWindowResponse> callback) {
            AlphaIntelligence.this.analyticsSuccessCallback = callback;
            return this;
        }

        /**
         * Sets the failure callback for asynchronous fetch.
         *
         * @param callback Failure callback (exception passed on failure)
         * @return this proxy for method chaining
         */
        public AnalyticsFixedWindowRequestProxy onFailure(FailureCallback callback) {
            AlphaIntelligence.this.analyticsFailureCallback = callback;
            return this;
        }

        /**
         * Triggers the asynchronous fetch of the analytics request.
         * Sets this builder as the active analytics builder and disables insider builder.
         * Invokes AlphaIntelligence.fetch(), which will dispatch the correct HTTP request.
         */
        public void fetch() {
            AlphaIntelligence.this.analyticsBuilder = this.builder;
            AlphaIntelligence.this.insiderBuilder = null;
            AlphaIntelligence.this.fetch();
        }

        /**
         * Internal use: sets the synchronous response.
         *
         * @param response The response to be set.
         */
        public void setSyncResponse(AnalyticsFixedWindowResponse response) {
            this.syncResponse = response;
        }

        /**
         * Synchronously fetches the analytics response, blocking until completion.
         * Any exceptions are thrown as AlphaVantageException.
         *
         * @return AnalyticsFixedWindowResponse for the request.
         * @throws AlphaVantageException if network or API error occurs.
         */
        public AnalyticsFixedWindowResponse fetchSync() throws AlphaVantageException {
            SuccessCallback<AnalyticsFixedWindowResponse> callback = this::setSyncResponse;
            AlphaIntelligence.this.analyticsBuilder = this.builder;
            AlphaIntelligence.this.insiderBuilder = null;
            AlphaIntelligence.this.fetchAnalyticsSync(callback);
            return this.syncResponse;
        }
    }

    /**
     * Fluent proxy for building and executing Insider Transactions requests.
     * <p>
     * Supports both asynchronous and synchronous usage.
     * Users interact with this class to set parameters, callbacks, and to trigger the request.
     */
    public class InsiderTransactionsRequestProxy {

        /**
         * The builder instance used to construct the insider transactions request.
         * Provides methods for specifying request parameters such as the stock symbol.
         */
        protected InsiderTransactionsRequest.Builder builder = new InsiderTransactionsRequest.Builder();

        /**
         * The response object returned by a synchronous fetch operation.
         * Set internally after fetchSync() is called.
         */
        protected InsiderTransactionsResponse syncResponse;

        /**
         * Sets the stock symbol for the insider transactions request.
         *
         * @param symbol The ticker symbol (e.g., "AAPL")
         * @return this proxy for method chaining (fluent API)
         */
        public InsiderTransactionsRequestProxy symbol(String symbol) {
            builder.symbol(symbol); // Delegate symbol setting to the request builder.
            return this;
        }

        /**
         * Sets the success callback for asynchronous fetch.
         *
         * @param callback Success callback to be executed when a response is received successfully.
         * @return this proxy for method chaining (fluent API)
         */
        public InsiderTransactionsRequestProxy onSuccess(SuccessCallback<InsiderTransactionsResponse> callback) {
            // Store callback in the parent AlphaIntelligence instance.
            AlphaIntelligence.this.insiderSuccessCallback = callback;
            return this;
        }

        /**
         * Sets the failure callback for asynchronous fetch.
         *
         * @param callback Failure callback to be executed when a request fails or API returns an error.
         * @return this proxy for method chaining (fluent API)
         */
        public InsiderTransactionsRequestProxy onFailure(FailureCallback callback) {
            // Store callback in the parent AlphaIntelligence instance.
            AlphaIntelligence.this.insiderFailureCallback = callback;
            return this;
        }

        /**
         * Triggers the asynchronous fetch of the insider transactions request.
         * <p>
         * Steps:
         * 1. Sets this builder as the active builder for insider requests.
         * 2. Clears the analytics builder (since only one request can be "active" at a time).
         * 3. Calls fetch() on the parent AlphaIntelligence instance, which dispatches the actual HTTP request.
         */
        public void fetch() {
            AlphaIntelligence.this.insiderBuilder = this.builder;
            AlphaIntelligence.this.analyticsBuilder = null;
            AlphaIntelligence.this.fetch();
        }

        /**
         * Internal use: sets the synchronous response.
         *
         * @param response The response to be set (used by sync callback).
         */
        public void setSyncResponse(InsiderTransactionsResponse response) {
            this.syncResponse = response;
        }

        /**
         * Synchronously fetches the insider transactions response, blocking until completion.
         * <p>
         * Flow:
         * 1. Builds and sets up the request.
         * 2. Calls the synchronous fetch method in AlphaIntelligence, passing a callback that sets syncResponse.
         * 3. Returns the syncResponse set by the callback.
         * <p>
         * Any exceptions are thrown as AlphaVantageException.
         *
         * @return InsiderTransactionsResponse for the request.
         * @throws AlphaVantageException if a network or API error occurs.
         */
        public InsiderTransactionsResponse fetchSync() throws AlphaVantageException {
            // Callback that simply stores the response for retrieval after blocking call.
            SuccessCallback<InsiderTransactionsResponse> callback = this::setSyncResponse;
            AlphaIntelligence.this.insiderBuilder = this.builder;
            AlphaIntelligence.this.analyticsBuilder = null;
            AlphaIntelligence.this.fetchInsiderSync(callback);
            return this.syncResponse;
        }
    }
}