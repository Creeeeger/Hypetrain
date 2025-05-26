package com.crazzyghost.alphavantage.realtime;

import com.crazzyghost.alphavantage.AlphaVantageException;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.Fetcher;
import com.crazzyghost.alphavantage.UrlExtractor;
import com.crazzyghost.alphavantage.parser.Parser;
import com.crazzyghost.alphavantage.realtime.request.RealTimeRequest;
import com.crazzyghost.alphavantage.realtime.response.RealTimeResponse;
import okhttp3.*;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;

/**
 * RealTime fetcher class for requesting real-time bulk stock quotes
 * from Alpha Vantage's API. Supports both synchronous and asynchronous
 * HTTP requests with callback support for handling success or failure.
 */
public final class RealTime implements Fetcher {

    // Configuration object holding API key and HTTP client instance
    private final Config config;

    // Builder object to construct the API request URL with parameters
    private final RealTimeRequest.Builder builder;

    // Callback to handle success events when using async fetch
    private SuccessCallback<RealTimeResponse> successCallback;

    // Callback to handle failure events when using async fetch
    private FailureCallback failureCallback;

    /**
     * Constructor initializes the RealTime fetcher with provided configuration.
     * The Config object contains API key and OkHttpClient instance.
     *
     * @param config configuration containing API key and HTTP client setup
     */
    public RealTime(Config config) {
        this.config = config;
        this.builder = new RealTimeRequest.Builder();
    }

    /**
     * Sets the symbol (stock tickers) for which to fetch real-time quotes.
     * The symbol should be provided as a comma-separated string, e.g. "AAPL,MSFT".
     *
     * @param symbol comma-separated list of stock symbol
     * @return the current RealTime instance to allow method chaining
     */
    public RealTime setSymbols(String symbol) {
        this.builder.symbol(symbol);
        return this;
    }

    // set realtime or delayed
    public RealTime entitlement(String entitlement) {
        this.builder.entitlement(entitlement);
        return this;
    }

    /**
     * Registers a callback to handle the response in case of a successful
     * asynchronous fetch. This callback will be invoked with the parsed
     * RealTimeResponse object when data is fetched successfully.
     *
     * @param callback success callback handler
     * @return the current RealTime instance to allow method chaining
     */
    public RealTime onSuccess(SuccessCallback<RealTimeResponse> callback) {
        this.successCallback = callback;
        return this;
    }

    /**
     * Registers a callback to handle any failure that occurs during
     * an asynchronous fetch operation. This could be network errors,
     * parsing errors, or API error messages.
     *
     * @param callback failure callback handler
     * @return the current RealTime instance to allow method chaining
     */
    public RealTime onFailure(FailureCallback callback) {
        this.failureCallback = callback;
        return this;
    }

    /**
     * Performs a synchronous (blocking) HTTP request to fetch real-time
     * stock quotes from the API.
     * <p>
     * This method blocks the calling thread until the response is received
     * or an exception occurs. The response body is parsed into a
     * RealTimeResponse object.
     *
     * @return parsed RealTimeResponse containing quote data or errors
     * @throws AlphaVantageException if a network, IO or parsing error occurs
     */
    public RealTimeResponse fetchSync() throws AlphaVantageException {
        // Validate configuration to ensure API key and HTTP client are present
        Config.checkNotNullOrKeyEmpty(config);

        // Clear any async callbacks since they are not used in sync fetch
        this.successCallback = null;
        this.failureCallback = null;

        // Retrieve the HTTP client from config
        OkHttpClient client = config.getOkHttpClient();

        // Build the API request URL with query parameters and API key
        Request request = UrlExtractor.extract(builder.build(), config.getKey());

        try (
                // Execute the HTTP request synchronously and obtain response
                Response response = client.newCall(request).execute()
        ) {
            // Extract response body string, or null if no body
            String bodyString = response.body() != null ? response.body().string() : null;

            // Parse JSON response into RealTimeResponse object and return
            return RealTimeResponse.of(Parser.parseJSON(bodyString));

        } catch (IOException e) {
            // Wrap IOExceptions in custom AlphaVantageException for caller handling
            throw new AlphaVantageException(e.getMessage());
        }
    }

    /**
     * Performs an asynchronous HTTP request to fetch real-time stock quotes.
     * <p>
     * On completion, either the registered successCallback or failureCallback
     * is invoked depending on whether the request and response parsing succeeded.
     * <p>
     * This method returns immediately without blocking.
     */
    @Override
    public void fetch() {
        // Validate config before starting request
        Config.checkNotNullOrKeyEmpty(config);

        // Build the API request URL
        Request request = UrlExtractor.extract(builder.build(), config.getKey());

        // Execute the HTTP request asynchronously using OkHttp's enqueue method
        config.getOkHttpClient().newCall(request).enqueue(new Callback() {

            /**
             * Called when the HTTP request fails (e.g., network error, timeout).
             *
             * @param call the HTTP call object
             * @param e exception describing failure cause
             */
            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                // If a failure callback is registered, invoke it with wrapped exception
                if (failureCallback != null) {
                    failureCallback.onFailure(new AlphaVantageException(e.getMessage()));
                }
            }

            /**
             * Called when the HTTP response is received from the API.
             *
             * @param call the HTTP call object
             * @param response the HTTP response object
             * @throws IOException if reading the response body fails
             */
            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    // Safely retrieve the response body (JSON string)
                    try (ResponseBody body = response.body()) {
                        String json = body != null ? body.string() : null;

                        // Parse the JSON into RealTimeResponse object
                        RealTimeResponse realTimeResponse = RealTimeResponse.of(Parser.parseJSON(json));

                        // Check if API returned an error message inside the JSON
                        if (realTimeResponse.getErrorMessage() != null && failureCallback != null) {
                            // Invoke failure callback with API error message
                            failureCallback.onFailure(new AlphaVantageException(realTimeResponse.getErrorMessage()));
                            return;
                        }

                        // If successful and success callback registered, invoke it
                        if (successCallback != null) {
                            successCallback.onSuccess(realTimeResponse);
                        }
                    }
                } else {
                    // HTTP response not successful (e.g., 4xx or 5xx status code)
                    if (failureCallback != null) {
                        failureCallback.onFailure(new AlphaVantageException("Request was unsuccessful"));
                    }
                }
            }
        });
    }
}