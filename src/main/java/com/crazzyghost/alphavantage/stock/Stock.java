package com.crazzyghost.alphavantage.stock;

import com.crazzyghost.alphavantage.AlphaVantageException;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.Fetcher;
import com.crazzyghost.alphavantage.UrlExtractor;
import com.crazzyghost.alphavantage.parser.Parser;
import com.crazzyghost.alphavantage.stock.request.StockRequest;
import com.crazzyghost.alphavantage.stock.response.StockResponse;
import okhttp3.*;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;

/**
 * Fetcher for stock symbol search using the Alpha Vantage API.
 * <p>
 * Supports synchronous and asynchronous (callback-based) execution.
 * You can set search keywords and register success/failure callbacks before fetching.
 * </p>
 * <p>
 * Usage:
 * <pre>
 *     Stock fetcher = new Stock(config).setKeywords("Tesla");
 *     StockResponse response = fetcher.fetchSync();
 *     // or
 *     fetcher.onSuccess(r -> { ... }).onFailure(e -> { ... }).fetch();
 * </pre>
 */
public final class Stock implements Fetcher {
    /**
     * Alpha Vantage API configuration.
     */
    private final Config config;

    /**
     * Builder for constructing the StockRequest.
     */
    private final StockRequest.Builder builder;

    /**
     * Callback for successful async responses.
     */
    private SuccessCallback<StockResponse> successCallback;

    /**
     * Callback for failed async responses.
     */
    private FailureCallback failureCallback;

    /**
     * Constructs a Stock fetcher with the given API configuration.
     * The request builder is initialized with default settings.
     *
     * @param config API configuration object (must have API key).
     */
    public Stock(Config config) {
        this.config = config;
        this.builder = new StockRequest.Builder();
    }

    /**
     * Sets the keywords to search for (company name, ticker, etc.).
     *
     * @param keywords Search string (e.g., "Apple").
     * @return This Stock instance for method chaining.
     */
    public Stock setKeywords(String keywords) {
        this.builder.forKeywords(keywords);
        return this;
    }

    /**
     * Registers a callback to be executed when a successful response is received (asynchronously).
     *
     * @param callback Success callback (consumes {@link StockResponse}).
     * @return This Stock instance for method chaining.
     */
    public Stock onSuccess(SuccessCallback<StockResponse> callback) {
        this.successCallback = callback;
        return this;
    }

    /**
     * Registers a callback to be executed when a failure or error occurs (asynchronously).
     *
     * @param callback Failure callback (consumes {@link AlphaVantageException}).
     * @return This Stock instance for method chaining.
     */
    public Stock onFailure(FailureCallback callback) {
        this.failureCallback = callback;
        return this;
    }

    /**
     * Executes the symbol search synchronously and returns the parsed response.
     * <p>
     * Any previous async callbacks are cleared. Throws an exception if there is an error.
     *
     * @return {@link StockResponse} containing matches and possible error message.
     * @throws AlphaVantageException If there is an error with the HTTP request or parsing.
     */
    public StockResponse fetchSync() throws AlphaVantageException {

        // Ensure API key and config are valid
        Config.checkNotNullOrKeyEmpty(config);

        // Clear any async callbacks to avoid confusion
        this.successCallback = null;
        this.failureCallback = null;
        OkHttpClient client = config.getOkHttpClient();

        try (Response response = client.newCall(
                UrlExtractor.extract(builder.build(), config.getKey())
        ).execute()) {

            // Parse the response body JSON to StockResponse
            return StockResponse.of(Parser.parseJSON(
                    response.body() != null ? response.body().string() : null
            ));
        } catch (IOException e) {
            throw new AlphaVantageException(e.getMessage());
        }
    }

    /**
     * Executes the symbol search asynchronously using registered callbacks.
     * <p>
     * The success and failure callbacks will be triggered depending on the outcome.
     */
    @Override
    public void fetch() {

        // Ensure API key and config are valid
        Config.checkNotNullOrKeyEmpty(config);

        // Execute request asynchronously using OkHttp
        config.getOkHttpClient().newCall(
                UrlExtractor.extract(builder.build(), config.getKey())
        ).enqueue(new Callback() {
            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                // If failure callback is set, call it with exception
                if (failureCallback != null) failureCallback.onFailure(new AlphaVantageException());
            }

            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    try (ResponseBody body = response.body()) {
                        // Parse the JSON response body into StockResponse
                        StockResponse stockResponse = StockResponse.of(Parser.parseJSON(
                                body != null ? body.string() : null
                        ));
                        // If parsing produced an error, treat as failure
                        if (stockResponse.getErrorMessage() != null && failureCallback != null) {
                            failureCallback.onFailure(new AlphaVantageException(stockResponse.getErrorMessage()));
                        }
                        // Otherwise, call the success callback with the response
                        if (successCallback != null) {
                            successCallback.onSuccess(stockResponse);
                        }
                    }
                } else {
                    // HTTP response was not successful (status code >=400), trigger failure callback
                    if (failureCallback != null) {
                        failureCallback.onFailure(new AlphaVantageException());
                    }
                }
            }
        });
    }
}
