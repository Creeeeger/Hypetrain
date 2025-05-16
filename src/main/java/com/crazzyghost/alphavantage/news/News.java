package com.crazzyghost.alphavantage.news;

import com.crazzyghost.alphavantage.AlphaVantageException;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.Fetcher;
import com.crazzyghost.alphavantage.UrlExtractor;
import com.crazzyghost.alphavantage.news.request.NewsRequest;
import com.crazzyghost.alphavantage.news.response.NewsResponse;
import com.crazzyghost.alphavantage.parser.Parser;
import okhttp3.*;
import org.jetbrains.annotations.NotNull;

import java.io.IOException;

/**
 * Handles fetching news sentiment data from Alpha Vantage API.
 * Supports both synchronous and asynchronous requests.
 * Uses the Builder pattern to configure request parameters.
 */
public final class News implements Fetcher {

    private final Config config;
    private final NewsRequest.Builder builder;
    private SuccessCallback<NewsResponse> successCallback;
    private FailureCallback failureCallback;

    /**
     * Creates a new News fetcher with the given configuration.
     *
     * @param config configuration including API key and HTTP client
     */
    public News(Config config) {
        this.config = config;
        this.builder = new NewsRequest.Builder();
    }

    /**
     * Sets the ticker symbols filter for the news request.
     *
     * @param tickers comma-separated ticker symbols (e.g., "AAPL,MSFT")
     * @return this News instance for chaining
     */
    public News setTickers(String tickers) {
        this.builder.tickers(tickers);
        return this;
    }

    /**
     * Sets the topics filter for the news request.
     *
     * @param topics comma-separated topic keywords
     * @return this News instance for chaining
     */
    public News setTopics(String topics) {
        this.builder.topics(topics);
        return this;
    }

    /**
     * Sets the start datetime for filtering news.
     *
     * @param timeFrom ISO 8601 datetime string
     * @return this News instance for chaining
     */
    public News setTimeFrom(String timeFrom) {
        this.builder.timeFrom(timeFrom);
        return this;
    }

    /**
     * Sets the end datetime for filtering news.
     *
     * @param timeTo ISO 8601 datetime string
     * @return this News instance for chaining
     */
    public News setTimeTo(String timeTo) {
        this.builder.timeTo(timeTo);
        return this;
    }

    /**
     * Sets the sort order of the news results.
     *
     * @param sort sorting option (e.g., "LATEST")
     * @return this News instance for chaining
     */
    public News setSort(String sort) {
        this.builder.sort(sort);
        return this;
    }

    /**
     * Sets the maximum number of news items to retrieve.
     *
     * @param limit maximum results limit
     * @return this News instance for chaining
     */
    public News setLimit(int limit) {
        this.builder.limit(limit);
        return this;
    }

    /**
     * Registers a callback to be invoked on successful async fetch.
     *
     * @param callback callback handling successful NewsResponse
     * @return this News instance for chaining
     */
    public News onSuccess(SuccessCallback<NewsResponse> callback) {
        this.successCallback = callback;
        return this;
    }

    /**
     * Registers a callback to be invoked on failure during async fetch.
     *
     * @param callback callback handling failure
     * @return this News instance for chaining
     */
    public News onFailure(FailureCallback callback) {
        this.failureCallback = callback;
        return this;
    }

    /**
     * Performs a synchronous API call to fetch news data.
     *
     * @return parsed NewsResponse from the API
     * @throws AlphaVantageException if network or parsing fails
     */
    public NewsResponse fetchSync() throws AlphaVantageException {
        // Validate config and API key presence
        Config.checkNotNullOrKeyEmpty(config);

        // Clear callbacks as they are not used in synchronous fetch
        this.successCallback = null;
        this.failureCallback = null;
        OkHttpClient client = config.getOkHttpClient();

        try (Response response = client.newCall(UrlExtractor.extract(builder.build(), config.getKey())).execute()) {
            // Parse the JSON response into a NewsResponse object
            String bodyString = response.body() != null ? response.body().string() : null;
            return NewsResponse.of(Parser.parseJSON(bodyString));
        } catch (IOException e) {
            throw new AlphaVantageException(e.getMessage());
        }
    }

    /**
     * Performs an asynchronous API call to fetch news data.
     * On completion, invokes registered success or failure callbacks.
     */
    @Override
    public void fetch() {
        // Validate config and API key presence
        Config.checkNotNullOrKeyEmpty(config);

        // Enqueue the HTTP call asynchronously
        config.getOkHttpClient().newCall(UrlExtractor.extract(builder.build(), config.getKey())).enqueue(new Callback() {

            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                // Invoke failure callback if registered, passing an exception wrapper
                if (failureCallback != null) failureCallback.onFailure(new AlphaVantageException());
            }

            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    try (ResponseBody body = response.body()) {
                        String json = body != null ? body.string() : null;
                        NewsResponse newsResponse = NewsResponse.of(Parser.parseJSON(json));

                        // If the response contains an error, treat as failure
                        if (newsResponse.getErrorMessage() != null && failureCallback != null) {
                            failureCallback.onFailure(new AlphaVantageException(newsResponse.getErrorMessage()));
                            return;
                        }

                        // Otherwise invoke success callback
                        if (successCallback != null) {
                            successCallback.onSuccess(newsResponse);
                        }
                    }
                } else {
                    // If HTTP response is unsuccessful, invoke failure callback
                    if (failureCallback != null) {
                        failureCallback.onFailure(new AlphaVantageException());
                    }
                }
            }
        });
    }
}