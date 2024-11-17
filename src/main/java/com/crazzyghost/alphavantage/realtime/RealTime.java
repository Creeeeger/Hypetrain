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

public final class RealTime implements Fetcher {
    private final Config config;
    private final RealTimeRequest.Builder builder;
    private SuccessCallback<RealTimeResponse> successCallback;
    private FailureCallback failureCallback;

    public RealTime(Config config) {
        this.config = config;
        this.builder = new RealTimeRequest.Builder();
    }

    public RealTime setSymbols(String symbols) {
        this.builder.symbols(symbols);
        return this;
    }

    public RealTime onSuccess(SuccessCallback<RealTimeResponse> callback) {
        this.successCallback = callback;
        return this;
    }

    public RealTime onFailure(FailureCallback callback) {
        this.failureCallback = callback;
        return this;
    }

    public RealTimeResponse fetchSync() throws AlphaVantageException {

        Config.checkNotNullOrKeyEmpty(config);

        this.successCallback = null;
        this.failureCallback = null;
        OkHttpClient client = config.getOkHttpClient();

        try (Response response = client.newCall(UrlExtractor.extract(builder.build(), config.getKey())).execute()) {
            return RealTimeResponse.of(Parser.parseJSON(response.body() != null ? response.body().string() : null));
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
                if (failureCallback != null) failureCallback.onFailure(new AlphaVantageException(e.getMessage()));
            }

            @Override
            public void onResponse(@NotNull Call call, @NotNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    try (ResponseBody body = response.body()) {
                        RealTimeResponse realTimeResponse = RealTimeResponse.of(Parser.parseJSON(body != null ? body.string() : null));
                        if (realTimeResponse.getErrorMessage() != null && failureCallback != null) {
                            failureCallback.onFailure(new AlphaVantageException(realTimeResponse.getErrorMessage()));
                        }
                        if (successCallback != null) {
                            successCallback.onSuccess(realTimeResponse);
                        }
                    }
                } else {
                    if (failureCallback != null) {
                        failureCallback.onFailure(new AlphaVantageException("Request was unsuccessful"));
                    }
                }
            }
        });
    }
}