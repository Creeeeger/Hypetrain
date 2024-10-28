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

public final class Stock implements Fetcher {
    private final Config config;
    private final StockRequest.Builder builder;
    private SuccessCallback<StockResponse> successCallback;
    private FailureCallback failureCallback;

    public Stock(Config config) {
        this.config = config;
        this.builder = new StockRequest.Builder();
    }

    public Stock setKeywords(String keywords) {
        this.builder.forKeywords(keywords);
        return this;
    }

    public Stock onSuccess(SuccessCallback<StockResponse> callback) {
        this.successCallback = callback;
        return this;
    }

    public Stock onFailure(FailureCallback callback) {
        this.failureCallback = callback;
        return this;
    }

    public StockResponse fetchSync() throws AlphaVantageException {

        Config.checkNotNullOrKeyEmpty(config);

        this.successCallback = null;
        this.failureCallback = null;
        OkHttpClient client = config.getOkHttpClient();

        try (Response response = client.newCall(UrlExtractor.extract(builder.build(), config.getKey())).execute()) {
            return StockResponse.of(Parser.parseJSON(response.body() != null ? response.body().string() : null));
        } catch (IOException e) {
            throw new AlphaVantageException(e.getMessage());
        }
    }

    @Override
    public void fetch() {

        Config.checkNotNullOrKeyEmpty(config);

        config.getOkHttpClient().newCall(UrlExtractor.extract(builder.build(), config.getKey())).enqueue(new Callback() {
            @Override
            public void onFailure(@NotNull Call call, @NotNull IOException e) {
                if (failureCallback != null) failureCallback.onFailure(new AlphaVantageException());
            }

            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    try (ResponseBody body = response.body()) {
                        StockResponse stockResponse = StockResponse.of(Parser.parseJSON(body != null ? body.string() : null));
                        if (stockResponse.getErrorMessage() != null && failureCallback != null) {
                            failureCallback.onFailure(new AlphaVantageException(stockResponse.getErrorMessage()));
                        }
                        if (successCallback != null) {
                            successCallback.onSuccess(stockResponse);
                        }
                    }
                } else {
                    if (failureCallback != null) {
                        failureCallback.onFailure(new AlphaVantageException());
                    }
                }
            }
        });
    }
}
