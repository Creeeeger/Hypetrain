package com.crazzyghost.alphavantage.realtime;

import com.crazzyghost.alphavantage.AlphaVantageException;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.Fetcher;
import com.crazzyghost.alphavantage.UrlExtractor;
import com.crazzyghost.alphavantage.parser.Parser;
import com.crazzyghost.alphavantage.realtime.request.RealTimeRequest;
import com.crazzyghost.alphavantage.realtime.response.RealTimeResponse;
import okhttp3.OkHttpClient;
import okhttp3.Response;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

import static org.crecker.mainDataHandler.stockSymbols;

public final class RealTime implements Fetcher {
    private final Config config;
    private final RealTimeRequest.Builder builder;
    private SuccessCallback<RealTimeResponse> successCallback;
    private FailureCallback failureCallback;
    private LocalDateTime currentTimestamp;
    private final Map<String, Double> symbolLastCloseMap;
    private final List<String> symbols;
    private final Random random = new Random();

    public RealTime(Config config) {
        this.config = config;
        this.builder = new RealTimeRequest.Builder();
        currentTimestamp = LocalDateTime.now();
        symbols = new ArrayList<>();
        symbolLastCloseMap = new HashMap<>();
        symbols.addAll(Arrays.asList(stockSymbols));
        for (String stockSymbol : stockSymbols) {
            symbolLastCloseMap.put(stockSymbol, 100.0); // Initial base price
        }
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

        // Increment timestamp by 1 minute each call
        currentTimestamp = currentTimestamp.plusSeconds(1);
        String timestampStr = currentTimestamp.format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS"));

        StringBuilder syntheticJsonBuilder = new StringBuilder();
        syntheticJsonBuilder.append("{\n")
                .append("  \"endpoint\": \"Realtime Bulk Quotes\",\n")
                .append("  \"message\": \"Synthetic test data for development purposes.\",\n")
                .append("  \"data\": [\n");

        // Generate data for each symbol
        for (int i = 0; i < symbols.size(); i++) {
            String symbol = symbols.get(i);
            double previousClose = symbolLastCloseMap.get(symbol);

            // Generate price changes (-2% to +2%)
            double changePercent = (random.nextDouble() * 4.0) - 2.0;
            double change = previousClose * changePercent / 100.0;
            double close = previousClose + change;

            // Generate open, high, low with variance
            double open = close * (0.99 + random.nextDouble() * 0.02);
            double high = close * (1.0 + random.nextDouble() * 0.01);
            double low = close * (0.99 - random.nextDouble() * 0.01);
            high = Math.max(high, Math.max(open, close));
            low = Math.min(low, Math.min(open, close));

            // Generate volume (1M to 50M)
            long volume = 1_000_000 + (long) (random.nextDouble() * 49_000_000);

            // Extended hours quote
            double extendedChangePercent = (random.nextDouble() * 0.5) - 0.25;
            double extendedQuote = close * (1 + extendedChangePercent / 100.0);
            double extendedChange = extendedQuote - close;
            double extendedChangePercentFormatted = (extendedChange / close) * 100;

            // Update last close
            symbolLastCloseMap.put(symbol, close);

            // Build JSON entry
            syntheticJsonBuilder.append("    {\n")
                    .append(String.format("      \"symbol\": \"%s\",\n", symbol))
                    .append(String.format("      \"timestamp\": \"%s\",\n", timestampStr))
                    .append(String.format("      \"open\": \"%.2f\",\n", open))
                    .append(String.format("      \"high\": \"%.2f\",\n", high))
                    .append(String.format("      \"low\": \"%.2f\",\n", low))
                    .append(String.format("      \"close\": \"%.2f\",\n", close))
                    .append(String.format("      \"volume\": \"%d\",\n", volume))
                    .append(String.format("      \"previous_close\": \"%.2f\",\n", previousClose))
                    .append(String.format("      \"change\": \"%.2f\",\n", change))
                    .append(String.format("      \"change_percent\": \"%.4f\",\n", changePercent))
                    .append(String.format("      \"extended_hours_quote\": \"%.2f\",\n", extendedQuote))
                    .append(String.format("      \"extended_hours_change\": \"%.2f\",\n", extendedChange))
                    .append(String.format("      \"extended_hours_change_percent\": \"%.4f\"\n", extendedChangePercentFormatted))
                    .append(i < symbols.size() - 1 ? "    },\n" : "    }\n");
        }

        syntheticJsonBuilder.append("  ]\n").append("}");

        RealTimeResponse realTimeResponse;
        try {
            realTimeResponse = RealTimeResponse.of(Parser.parseJSON(syntheticJsonBuilder.toString()));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        if (successCallback != null) {
            successCallback.onSuccess(realTimeResponse);
        }
    }
}