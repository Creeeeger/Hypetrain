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

public final class RealTime implements Fetcher {
    private final Config config;
    private final RealTimeRequest.Builder builder;
    private SuccessCallback<RealTimeResponse> successCallback;
    private FailureCallback failureCallback;
    private LocalDateTime currentTimestamp;
    private final Map<String, Double> symbolLastCloseMap;
    private final List<String> symbols;
    private final Random random = new Random();

    String[] stockSymbols = {
            "1Q", "AAOI", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACHR", "ADBE", "ADI", "ADP", "ADSK", "AEM", "AER", "AES", "AFL", "AFRM", "AJG", "AKAM", "ALAB"
            , "AMAT", "AMC", "AMD", "AME", "AMGN", "AMT", "AMZN", "ANET", "AON", "AOSL", "APD", "APH", "APLD", "APO", "APP", "APTV", "ARE", "ARM", "ARWR", "AS"
                , "ASML", "ASPI", "ASTS", "AVGO", "AXP", "AZN", "AZO", "Al", "BA", "BABA", "BAC", "BBY", "BDX", "BE", "BKNG", "BKR", "BLK", "BMO", "BMRN", "BMY"
                , "BN", "BNS", "BNTX", "BP", "BRK/B", "BSX", "BTDR", "BTI", "BUD", "BX", "C", "CARR", "CAT", "CAVA", "CB", "CBRE", "CDNS", "CEG", "CELH", "CF"
                , "CI", "CIFR", "CLSK", "CLX", "CMCSA", "CME", "CMG", "CNI", "CNQ", "COF", "COHR", "COIN", "COP", "CORZ", "COST", "CP", "CRDO", "CRM", "CRWD", "CRWV"
                , "CSCO", "CSX", "CTAS", "CTVA", "CVNA", "CVS", "DAVE", "DDOG", "DE", "DEO", "DFS", "DGX", "DHI", "DHR", "DIS", "DJT", "DKNG", "DOCU", "DUK"
                , "DUOL", "DXYZ", "EA", "ECL", "ELF", "ELV", "ENB", "ENPH", "EOG", "EPD", "EQIX", "EQNR", "ET", "EW", "EXAS", "EXPE", "FCX", "FDX", "FERG", "FI"
                , "FIVE", "FLNC", "FMX", "FN", "FSLR", "FTAI", "FTNT", "FUTU", "GD", "GE", "GEV", "GGG", "GILD", "GIS", "GLW", "GM", "GMAB", "GME", "GOOGL", "GS"
                , "GSK", "GWW", "HCA", "HD", "HDB", "HES", "HIMS", "HON", "HOOD", "HSAI", "HSBC", "HSY", "HUT", "IBM", "IBN", "ICE", "IDXX", "IESC", "INFY", "INOD"
                , "INSP", "INTC", "INTU", "IONQ", "IREN", "IRM", "ISRG", "IT", "ITW", "JD", "JOBY", "JPM", "KHC", "KKR", "KLAC", "KODK", "LCID", "LIN"
                , "LKNC", "LLY", "LMND", "LMT", "LNG", "LNTH", "LOW", "LPLA", "LRCX", "LULU", "LUMN", "LUNR", "LUV", "LVS", "LX", "MA", "MAR", "MARA", "MBLY"
                , "MCHP", "MCK", "MCO", "MDB", "MDGL", "MDLZ", "MDT", "MET", "META", "MGM", "MKC", "MMC", "MMM", "MO", "MPWR", "MRK", "MRNA", "MRVL", "MS", "MSFT"
                , "MSI", "MSTR", "MT", "MU", "MUFG", "NFE", "NFLX", "NGG", "NIO", "NKE", "NNE", "NOC", "NOVA", "NOW", "NSC", "NVDA", "NVO", "NVS", "NXPI"
                , "O", "ODFL", "OKE", "OKLO", "OMC", "OPEN", "ORCL", "ORLY", "PANW", "PBR", "PCG", "PDD", "PFG", "PGHL", "PGR", "PH", "PLD"
                , "PLTR", "PLUG", "PM", "PNC", "POOL", "POWL", "PSA", "PSX", "PTON", "PYPL", "QBTS", "QCOM", "QUBT", "RACE", "RCAT", "RDDT", "REG", "REGN", "RELX", "RGTI"
                , "RIO", "RIOT", "RIVN", "RKLB", "ROOT", "ROP", "RSG", "RTX", "RUN", "RXRX", "RY", "SAP", "SBUX", "SCCO", "SCHW", "SE", "SEDG", "SG", "SHOP", "SHW"
                , "SLB", "SMCI", "SMFG", "SMLR", "SMR", "SMTC", "SNOW", "SNPS", "SNY", "SOFI", "SONY", "SOUN", "SPGI", "SPOT", "STRL", "SWK", "SWKS", "SYK", "SYM"
                , "SYY", "TCOM", "TD", "TDG", "TEM", "TFC", "TGT", "TJX", "TM", "TMDX", "TMO", "TMUS", "TRI", "TRU", "TRV", "TSLA", "TSN", "TT"
                , "TTD", "TTE", "TTEK", "TXN", "TXRH", "U", "UBER", "UBS", "UL", "ULTA", "UNH", "UNP", "UPS", "UPST", "URI", "USB", "USFD", "UTHR", "V", "VKTX"
                , "VLO", "VRSK", "VRSN", "VRT", "VRTX", "VST", "W", "WDAY", "WELL", "WFC", "WM", "WOLF", "WULF", "XOM", "XPEV", "XPO", "YUM", "ZETA"
            , "ZIM", "ZTO", "ZTS", "ВТВТ"
    };

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