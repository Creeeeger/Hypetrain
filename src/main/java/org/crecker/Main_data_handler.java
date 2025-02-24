package org.crecker;

import com.crazzyghost.alphavantage.AlphaVantage;
import com.crazzyghost.alphavantage.AlphaVantageException;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.fundamentaldata.response.CompanyOverview;
import com.crazzyghost.alphavantage.fundamentaldata.response.CompanyOverviewResponse;
import com.crazzyghost.alphavantage.news.response.NewsResponse;
import com.crazzyghost.alphavantage.parameters.Interval;
import com.crazzyghost.alphavantage.parameters.OutputSize;
import com.crazzyghost.alphavantage.realtime.response.RealTimeResponse;
import com.crazzyghost.alphavantage.stock.response.StockResponse;
import com.crazzyghost.alphavantage.timeseries.response.QuoteResponse;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import com.crazzyghost.alphavantage.timeseries.response.TimeSeriesResponse;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;

import java.awt.*;
import java.io.*;
import java.time.DayOfWeek;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.crecker.Main_UI.logTextArea;
import static org.crecker.RallyPredictor.predict;
import static org.crecker.csvDataGen.saveFeaturesToCSV;

public class Main_data_handler {

    public static final Map<String, Map<String, Double>> INDICATOR_RANGE_MAP = new LinkedHashMap<>() {{
        // Trend Following Indicators
        put("SMA_CROSS", Map.of("min", 0.0, "max", 1.0));
        put("EMA_CROSS", Map.of("min", 0.0, "max", 1.0));
        put("PRICE_SMA_DISTANCE", Map.of("min", 0.0, "max", 1.0));
        put("MACD", Map.of("min", -5.0, "max", 5.0));
        put("TRIX", Map.of("min", -20.0, "max", 20.0));
        put("KAMA", Map.of("min", 0.0, "max", 100.0));

        // Momentum Indicators
        put("RSI", Map.of("min", 0.0, "max", 100.0));
        put("ROC", Map.of("min", -30.0, "max", 30.0));
        put("MOMENTUM", Map.of("min", -10.0, "max", 10.0));
        put("CMO", Map.of("min", -100.0, "max", 100.0));
        put("ACCELERATION", Map.of("min", -1.0, "max", 1.0));

        // Volatility & Breakouts Indicators
        put("BOLLINGER", Map.of("min", 0.0, "max", 1.0));
        put("BREAKOUT_RESISTANCE", Map.of("min", 0.0, "max", 1.0));
        put("DONCHIAN", Map.of("min", 0.0, "max", 1.0));
        put("VOLATILITY_THRESHOLD", Map.of("min", 0.0, "max", 1.0));
        put("VOLATILITY_MONITOR", Map.of("min", 0.0, "max", 20.0));

        // Patterns Indicators
        put("CONSECUTIVE_POSITIVE_CLOSES", Map.of("min", 0.0, "max", 50.0));
        put("HIGHER_HIGHS", Map.of("min", 0.0, "max", 1.0));
        put("FRACTAL_BREAKOUT", Map.of("min", 0.0, "max", 1.0));
        put("CANDLE_PATTERN", Map.of("min", 0.0, "max", 5.0));
        put("TRENDLINE", Map.of("min", 0.0, "max", 1.0));

        // Statistical Indicators
        put("Z_SCORE", Map.of("min", 0.0, "max", 1.0));
        put("CUMULATIVE_PERCENTAGE", Map.of("min", 0.0, "max", 1.0));
        put("CUMULATIVE_THRESHOLD", Map.of("min", -20.0, "max", 20.0));

        // Advanced Indicators
        put("BREAKOUT_MA", Map.of("min", 0.0, "max", 1.0));
        put("PARABOLIC", Map.of("min", 0.0, "max", 1.0));
        put("KELTNER", Map.of("min", 0.0, "max", 1.0));
        put("ELDER_RAY", Map.of("min", -10.0, "max", 10.0));
        put("VOLUME_SPIKE", Map.of("min", 0.0, "max", 1.0));
        put("ATR", Map.of("min", 0.0, "max", 5.0));
    }};

    // Trend Following Indicators
    public static final Map<String, Double> TREND_FOLLOWING_WEIGHTS = Map.ofEntries(
            Map.entry("SMA_CROSS", 0.15),
            Map.entry("EMA_CROSS", 0.15),
            Map.entry("PRICE_SMA_DISTANCE", 0.20),
            Map.entry("MACD", 0.15),
            Map.entry("TRIX", 0.20),
            Map.entry("KAMA", 0.15)
    );

    // Momentum Indicators
    public static final Map<String, Double> MOMENTUM_WEIGHTS = Map.ofEntries(
            Map.entry("RSI", 0.25),
            Map.entry("ROC", 0.15),
            Map.entry("MOMENTUM", 0.15),
            Map.entry("CMO", 0.15),
            Map.entry("ACCELERATION", 0.30)
    );

    // Volatility & Breakouts Indicators
    public static final Map<String, Double> VOLATILITY_BREAKOUTS_WEIGHTS = Map.ofEntries(
            Map.entry("BOLLINGER", 0.20),
            Map.entry("BREAKOUT_RESISTANCE", 0.15),
            Map.entry("DONCHIAN", 0.20),
            Map.entry("VOLATILITY_THRESHOLD", 0.20),
            Map.entry("VOLATILITY_MONITOR", 0.25)
    );

    // Patterns Indicators
    public static final Map<String, Double> PATTERNS_WEIGHTS = Map.ofEntries(
            Map.entry("CONSECUTIVE_POSITIVE_CLOSES", 0.25),
            Map.entry("HIGHER_HIGHS", 0.15),
            Map.entry("FRACTAL_BREAKOUT", 0.20),
            Map.entry("CANDLE_PATTERN", 0.15),
            Map.entry("TRENDLINE", 0.25)
    );

    // Statistical Indicators
    public static final Map<String, Double> STATISTICAL_WEIGHTS = Map.ofEntries(
            Map.entry("Z_SCORE", 0.40),
            Map.entry("CUMULATIVE_PERCENTAGE", 0.40),
            Map.entry("CUMULATIVE_THRESHOLD", 0.20)
    );

    // Advanced Indicators
    public static final Map<String, Double> ADVANCED_WEIGHTS = Map.ofEntries(
            Map.entry("BREAKOUT_MA", 0.10),
            Map.entry("PARABOLIC", 0.10),
            Map.entry("KELTNER", 0.20),
            Map.entry("ELDER_RAY", 0.20),
            Map.entry("VOLUME_SPIKE", 0.25),
            Map.entry("ATR", 0.15)
    );

    // Aggregated weights map
    public static final Map<String, Map<String, Double>> INDICATOR_RANGE_FULL = Map.of(
            "TrendFollowing", TREND_FOLLOWING_WEIGHTS,
            "Momentum", MOMENTUM_WEIGHTS,
            "VolatilityBreakouts", VOLATILITY_BREAKOUTS_WEIGHTS,
            "Patterns", PATTERNS_WEIGHTS,
            "Statistical", STATISTICAL_WEIGHTS,
            "Advanced", ADVANCED_WEIGHTS
    );

    // Category Level Weights
    public static final Map<String, Double> INDICATOR_WEIGHTS_FULL = Map.of(
            "TrendFollowing", 0.20,
            "Momentum", 0.15,
            "VolatilityBreakouts", 0.25,
            "Patterns", 0.15,
            "Statistical", 0.10,
            "Advanced", 0.15
    );

    private static final Map<String, DoubleArrayWindow> volatilityWindows = new ConcurrentHashMap<>();
    private static final Map<String, DoubleArrayWindow> returnsWindows = new ConcurrentHashMap<>();
    public static Map<String, List<StockUnit>> symbolTimelines = new HashMap<>();
    public static int frameSize = 30; // Frame size for analysis
    public static List<Notification> notificationsForPLAnalysis = new ArrayList<>();
    public static boolean test = true; //if True use demo url for real Time Updates

    public static void InitAPi(String token) {
        // Configure the API client
        Config cfg = Config.builder()
                .key(token)
                .timeOut(10) // Timeout in seconds
                .build();

        // Initialize the Alpha Vantage API
        AlphaVantage.api().init(cfg);
    }

    public static void get_timeline(String symbol_name, TimelineCallback callback) {
        List<StockUnit> stocks = new ArrayList<>(); // Directly use a List<StockUnit>

        AlphaVantage.api()
                .timeSeries()
                .intraday()
                .forSymbol(symbol_name)
                .interval(Interval.ONE_MIN)
                .outputSize(OutputSize.FULL)
                .onSuccess(e -> {
                    TimeSeriesResponse response = (TimeSeriesResponse) e;
                    stocks.addAll(response.getStockUnits()); // Populate the list
                    callback.onTimeLineFetched(stocks); // Call the callback with the Stock list
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void get_Info_Array(String symbol_name, DataCallback callback) {
        Double[] data = new Double[9];

        // Fetch fundamental data
        AlphaVantage.api()
                .fundamentalData()
                .companyOverview()
                .forSymbol(symbol_name)
                .onSuccess(e -> {
                    CompanyOverviewResponse overview_response = (CompanyOverviewResponse) e;
                    CompanyOverview response = overview_response.getOverview();
                    data[4] = response.getPERatio();
                    data[5] = response.getPEGRatio();
                    data[6] = response.getFiftyTwoWeekHigh();
                    data[7] = response.getFiftyTwoWeekLow();
                    data[8] = Double.valueOf(response.getMarketCapitalization());

                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();

        AlphaVantage.api()
                .timeSeries()
                .quote()
                .forSymbol(symbol_name)
                .onSuccess(e -> {
                    QuoteResponse response = (QuoteResponse) e;
                    data[0] = response.getOpen();
                    data[1] = response.getHigh();
                    data[2] = response.getLow();
                    data[3] = response.getVolume();

                    // Call the callback with the fetched data
                    callback.onDataFetched(data);
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void handleFailure(AlphaVantageException error) {
        error.printStackTrace();
    }

    public static void findMatchingSymbols(String searchText, SymbolSearchCallback callback) {
        AlphaVantage.api()
                .Stocks()
                .setKeywords(searchText)
                .onSuccess(e -> {
                    List<String> allSymbols = e.getMatches()
                            .parallelStream()
                            .map(StockResponse.StockMatch::getSymbol)
                            .toList();

                    // Perform parallel filtering and pass result to callback
                    List<String> filteredSymbols = allSymbols.parallelStream()
                            .filter(symbol -> symbol.toUpperCase().startsWith(searchText.toUpperCase()))
                            .collect(Collectors.toList());
                    callback.onSuccess(filteredSymbols);
                })
                .onFailure(failure -> {
                    // Handle failure and invoke the failure callback
                    Main_data_handler.handleFailure(failure);
                    callback.onFailure(new RuntimeException("API call failed"));
                })
                .fetch();
    }

    public static void receive_News(String Symbol, ReceiveNewsCallback callback) {
        AlphaVantage.api()
                .News()
                .setTickers(Symbol)
                .setSort("LATEST")
                .setLimit(12)
                .onSuccess(e -> callback.onNewsReceived(e.getNewsItems()))
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void start_Hype_Mode(int tradeVolume, float hypeStrength) {
        String[] stockSymbols = {
                "1Q", "AAOI", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACHR", "ADBE", "ADI", "ADP", "ADSK", "AEM", "AER", "AES", "AFL", "AFRM", "AJG", "AKAM", "ALAB"
                , "AMAT", "AMC", "AMD", "AME", "AMGN", "AMT", "AMZN", "ANET", "AON", "AOSL", "APD", "APH", "APLD", "APO", "APP", "APTV", "ARE", "ARM", "ARWR", "AS"
                , "ASML", "ASPI", "ASTS", "AVGO", "AXP", "AZN", "AZO", "Al", "BA", "BABA", "BAC", "BBY", "BDX", "BE", "BKNG", "BKR", "BLK", "BMO", "BMRN", "BMY"
                , "BN", "BNS", "BNTX", "BP", "BRK/B", "BSX", "BTDR", "BTI", "BUD", "BX", "C", "CARR", "CAT", "CAVA", "CB", "CBRE", "CDNS", "CEG", "CELH", "CF"
                , "CI", "CIFR", "CLSK", "CLX", "CMCSA", "CME", "CMG", "CNI", "CNQ", "COF", "COHR", "COIN", "COP", "CORZ", "COST", "CP", "CRDO", "CRM", "CRWD"
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

        logTextArea.append(String.format("Activating hype mode for auto Stock scanning, Settings: %s Volume, %s Hype, %s Stocks to scan\n", tradeVolume, hypeStrength, stockSymbols.length));
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

        List<String> possibleSymbols = new ArrayList<>(); //get the symbols based on the config

        if (tradeVolume > 300000) {
            File file = new File(tradeVolume + ".txt");

            if (file.exists()) {
                // Load symbols from file if it exists
                try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        possibleSymbols.add(line);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }

                logTextArea.append("Loaded symbols from file\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                hypeModeFinder(possibleSymbols);
            } else {
                logTextArea.append("Started getting possible symbols\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                // If file does not exist, call API and write symbols to file
                get_available_symbols(tradeVolume, stockSymbols, result -> {
                    try (FileWriter writer = new FileWriter(file)) {
                        for (String s : result) {
                            String symbol = s.toUpperCase();
                            possibleSymbols.add(symbol);  // Add to possibleSymbols
                            writer.write(symbol + System.lineSeparator());  // Write to file
                        }
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    logTextArea.append("Finished getting possible symbols\n");
                    logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                    hypeModeFinder(possibleSymbols);
                });
            }
        } else {
            // For tradeVolume <= 300000, directly copy symbols to the list and process
            possibleSymbols.addAll(Arrays.asList(stockSymbols));

            logTextArea.append("Use pre set symbols\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

            hypeModeFinder(possibleSymbols);
        }
    }

    public static void get_available_symbols(int tradeVolume, String[] possibleSymbols, SymbolCallback callback) {
        List<String> actualSymbols = new ArrayList<>();

        for (int i = 0; i < possibleSymbols.length; i++) {
            int finalI = i;
            AlphaVantage.api()
                    .fundamentalData()
                    .companyOverview()
                    .forSymbol(possibleSymbols[i])
                    .onSuccess(e -> {
                        long market_cap = ((CompanyOverviewResponse) e).getOverview().getMarketCapitalization();
                        long outstanding_Shares = ((CompanyOverviewResponse) e).getOverview().getSharesOutstanding();

                        AlphaVantage.api()
                                .timeSeries()
                                .daily()
                                .forSymbol(possibleSymbols[finalI])
                                .outputSize(OutputSize.COMPACT)
                                .onSuccess(tsResponse -> {
                                    double close = ((TimeSeriesResponse) tsResponse).getStockUnits().get(0).getClose();
                                    double volume = ((TimeSeriesResponse) tsResponse).getStockUnits().get(0).getVolume();

                                    // Check conditions and add to actual symbols
                                    if (tradeVolume < market_cap) {
                                        if (((double) tradeVolume / close) < volume) {
                                            if (((long) tradeVolume / close) < outstanding_Shares) {
                                                actualSymbols.add(possibleSymbols[finalI]);
                                            }
                                        }
                                    }

                                    // Check if all symbols have been processed
                                    if (finalI == possibleSymbols.length - 1) {
                                        callback.onSymbolsAvailable(actualSymbols); // Call the callback when all are done
                                    }
                                })
                                .onFailure(Main_data_handler::handleFailure)
                                .fetch();
                    })
                    .onFailure(Main_data_handler::handleFailure)
                    .fetch();
        }
    }

    public static void hypeModeFinder(List<String> symbols) {
        logTextArea.append("Started pulling data from server\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

        // Use a thread-safe list for matches
        List<RealTimeResponse.RealTimeMatch> matches = Collections.synchronizedList(new ArrayList<>());

        while (!Thread.currentThread().isInterrupted()) {
            try {
                matches.clear();
                int totalBatches = (int) Math.ceil(symbols.size() / 100.0);
                CountDownLatch latch = new CountDownLatch(totalBatches);

                for (int i = 0; i < totalBatches; i++) {
                    List<String> batchSymbols = symbols.subList(i * 100, Math.min((i + 1) * 100, symbols.size()));
                    String symbolsBatch = String.join(",", batchSymbols).toUpperCase();

                    AlphaVantage.api()
                            .Realtime()
                            .setSymbols(symbolsBatch)
                            .onSuccess(response -> {
                                matches.addAll(response.getMatches());
                                latch.countDown();
                            })
                            .onFailure(e -> {
                                handleFailure(e);
                                latch.countDown(); // Ensure latch is counted down on failure
                            })
                            .fetch();
                }

                // Wait with timeout to prevent hanging
                if (!latch.await(5, TimeUnit.SECONDS)) {
                    logTextArea.append("Warning: Timed out waiting for data\n");
                }

                processStockData(matches);
                Thread.sleep(5000);

            } catch (InterruptedException e) {
                e.printStackTrace();
                Thread.currentThread().interrupt();
                logTextArea.append("Data pull interrupted\n");
                break;
            } catch (Exception e) {
                e.printStackTrace();
                logTextArea.append("Error during data pull: " + e.getMessage() + "\n");
            }
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        }
    }

    public static void getRealTimeUpdate(String symbol, RealTimeCallback callback) {
        AlphaVantage.api()
                .Realtime()
                .setSymbols(symbol)
                .onSuccess(response -> callback.onRealTimeReceived(response.getMatches().get(0)))
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void processStockData(List<RealTimeResponse.RealTimeMatch> matches) {
        Map<String, StockUnit> currentBatch = new HashMap<>();

        for (RealTimeResponse.RealTimeMatch match : matches) {
            String symbol = match.getSymbol().toUpperCase();
            StockUnit unit = new StockUnit.Builder()
                    .symbol(symbol)
                    .close(match.getClose())
                    .time(match.getTimestamp())
                    .volume(match.getVolume())
                    .build();

            // Update symbol timeline
            symbolTimelines.computeIfAbsent(symbol, k -> new ArrayList<>()).add(unit);
            currentBatch.put(symbol, unit);
        }
        logTextArea.append("Processed " + currentBatch.size() + " valid stock entries\n");
        calculateStockPercentageChange();
    }

    public static void calculateStockPercentageChange() {
        synchronized (symbolTimelines) {
            symbolTimelines.forEach((symbol, timeline) -> {
                if (timeline.size() < 2) {
                    logTextArea.append("Not enough data for " + symbol + "\n");
                    return;
                }

                int updates = 0;
                for (int i = 1; i < timeline.size(); i++) {
                    StockUnit current = timeline.get(i);
                    StockUnit previous = timeline.get(i - 1);

                    if (previous.getClose() > 0) {
                        double change = ((current.getClose() - previous.getClose()) / previous.getClose()) * 100;
                        change = Math.abs(change) >= 14 ? previous.getPercentageChange() : change;
                        current.setPercentageChange(change);
                        updates++;
                    }
                }

                logTextArea.append(symbol + ": Updated " + updates + " percentage changes\n");
            });
        }

        calculateSpikesInRally();
    }

    public static void calculateSpikesInRally() {
        rallyDetector(frameSize, true);
        checkToClean();
    }

    public static void checkToClean() {
        // Memory calculation remains the same
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        long usedMemoryInMB = usedMemory / (1024 * 1024);
        System.out.println("Used memory: " + usedMemoryInMB + " MB");

        if (usedMemoryInMB > 500) {
            synchronized (symbolTimelines) {
                final int MAX_ENTRIES = 1000;  // Keep last 1000 entries per symbol
                final int MIN_ENTRIES = 200;   // Minimum to maintain for analysis

                symbolTimelines.replaceAll((symbol, timeline) -> {
                    if (timeline.size() > MAX_ENTRIES) {
                        int removeCount = timeline.size() - MIN_ENTRIES;
                        return timeline.subList(removeCount, timeline.size());
                    }
                    return timeline;
                });

                // Optional: Remove empty entries
                symbolTimelines.entrySet().removeIf(entry -> entry.getValue().isEmpty());
            }

            System.out.println("Cleaned symbol timelines. Current counts:");
            symbolTimelines.forEach((sym, list) ->
                    System.out.println(sym + ": " + list.size() + " entries")
            );
        }
    }

    public static void rallyDetector(int minutesPeriod, boolean realFrame) {
        symbolTimelines.keySet()
                .parallelStream()
                .forEach(symbol -> {
                    List<StockUnit> timeline = getSymbolTimeline(symbol);
                    if (!timeline.isEmpty()) {
                        processTimeWindows(symbol, timeline, minutesPeriod, realFrame);
                    }
                });

        sortNotifications(notificationsForPLAnalysis);
    }

    private static void processTimeWindows(String symbol, List<StockUnit> timeline, int minutes, boolean useRealFrame) {
        List<Notification> stockNotifications = new ArrayList<>();

        if (useRealFrame) {
            // Process only the last relevant timeframe
            if (!timeline.isEmpty()) {
                LocalDateTime endTime = timeline.get(timeline.size() - 1).getLocalDateTimeDate();
                LocalDateTime startTime = endTime.minusMinutes(minutes);

                List<StockUnit> timeWindow = getTimeWindow(timeline, startTime, endTime);

                if (timeWindow.size() >= frameSize) {
                    try {
                        List<Notification> notifications = getNotificationForFrame(timeWindow, symbol);
                        stockNotifications.addAll(notifications);
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        } else {
            // Original notification-based processing
            timeline.parallelStream()
                    .forEach(stockUnit -> {
                        LocalDateTime startTime = stockUnit.getLocalDateTimeDate();
                        LocalDateTime endTime = startTime.plusMinutes(minutes);

                        List<StockUnit> timeWindow = getTimeWindow(timeline, startTime, endTime);

                        if (timeWindow.size() >= frameSize) {
                            try {
                                List<Notification> notifications = getNotificationForFrame(timeWindow, symbol);
                                stockNotifications.addAll(notifications);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    });
        }

        synchronized (notificationsForPLAnalysis) {
            notificationsForPLAnalysis.addAll(stockNotifications);
        }
    }

    private static List<StockUnit> getTimeWindow(List<StockUnit> timeline, LocalDateTime start, LocalDateTime end) {
        int startIndex = findTimeIndex(timeline, start);
        if (startIndex == -1) return Collections.emptyList();

        int endIndex = startIndex;
        while (endIndex < timeline.size() && !timeline.get(endIndex).getLocalDateTimeDate().isAfter(end)) {
            endIndex++;
        }

        return timeline.subList(startIndex, endIndex);
    }

    private static int findTimeIndex(List<StockUnit> timeline, LocalDateTime target) {
        int low = 0;
        int high = timeline.size() - 1;

        while (low <= high) {
            int mid = (low + high) / 2;
            LocalDateTime midTime = timeline.get(mid).getLocalDateTimeDate();

            if (midTime.isBefore(target)) {
                low = mid + 1;
            } else if (midTime.isAfter(target)) {
                high = mid - 1;
            } else {
                return mid;  // Exact match
            }
        }

        return low < timeline.size() ? low : -1;  // Nearest index if exact not found
    }

    public static void sortNotifications(List<Notification> notifications) {
        // Sort notifications by their time series end date
        notifications.sort((n1, n2) -> {
            LocalDateTime date1 = n1.getLocalDateTime();
            LocalDateTime date2 = n2.getLocalDateTime();
            return date1.compareTo(date2); // Sort from old to new
        });
    }

    public static double normalizeScore(String indicator, double rawValue) {
        Map<String, Double> range = INDICATOR_RANGE_MAP.get(indicator);
        if (range == null) return rawValue; // No normalization if indicator not found

        double min = range.get("min");
        double max = range.get("max");

        // Avoid division by zero in case of incorrect ranges
        if (max == min) return 0.0;

        return switch (indicator) {
            // Binary indicators: 0 if rawValue < 0.5, otherwise 1
            case "SMA_CROSS", "EMA_CROSS", "PRICE_SMA_DISTANCE", "BREAKOUT_RESISTANCE", "DONCHIAN",
                 "VOLATILITY_THRESHOLD", "HIGHER_HIGHS", "FRACTAL_BREAKOUT", "TRENDLINE", "Z_SCORE",
                 "CUMULATIVE_PERCENTAGE", "BREAKOUT_MA", "PARABOLIC", "KELTNER", "VOLUME_SPIKE" ->
                    rawValue >= 0.5 ? 1.0 : 0.0;

            // Continuous indicators with linear normalization to [0, 1]
            case "MACD", "TRIX", "KAMA", "RSI", "ROC", "MOMENTUM", "CMO", "ACCELERATION", "BOLLINGER",
                 "VOLATILITY_MONITOR", "CONSECUTIVE_POSITIVE_CLOSES", "CANDLE_PATTERN", "CUMULATIVE_THRESHOLD",
                 "ELDER_RAY", "ATR" -> {
                double shiftedValue = (rawValue - min) / (max - min);
                yield Math.max(0.0, Math.min(1.0, shiftedValue));
            }
            default -> Math.max(0.0, Math.min(1.0, (rawValue - min) / (max - min)));
        };
    }

    private static float[] computeFeatures(List<StockUnit> stocks, String symbol) {
        // Initialize feature array
        double[] features = new double[INDICATOR_RANGE_MAP.size()];
        int featureIndex = 0;

        // Trend Following Indicators
        features[featureIndex++] = normalizeScore("SMA_CROSS", normalizeScore("SMA_CROSS", isSMACrossover(stocks, 5, 20)));
        features[featureIndex++] = normalizeScore("EMA_CROSS", normalizeScore("EMA_CROSS", isEMACrossover(stocks, 10, 30)));
        features[featureIndex++] = normalizeScore("PRICE_SMA_DISTANCE", isPriceAboveSMA(stocks, 20));
        features[featureIndex++] = normalizeScore("MACD", calculateMACD(stocks).get("histogram"));
        features[featureIndex++] = normalizeScore("TRIX", calculateTRIX(stocks, 10));
        features[featureIndex++] = normalizeScore("KAMA", calculateKAMA(stocks, 10));

        // Momentum Indicators
        features[featureIndex++] = normalizeScore("RSI", calculateRSI(stocks, 14));
        features[featureIndex++] = normalizeScore("ROC", calculateROC(stocks, 10));
        features[featureIndex++] = normalizeScore("MOMENTUM", calculateMomentum(stocks, 10));
        features[featureIndex++] = normalizeScore("CMO", calculateCMO(stocks, 14));
        features[featureIndex++] = normalizeScore("ACCELERATION", calculateAcceleration(stocks, 8));

        // Volatility & Breakouts Indicators
        features[featureIndex++] = normalizeScore("BOLLINGER", calculateBollingerBands(stocks, 10).get("bandwidth"));
        features[featureIndex++] = normalizeScore("BREAKOUT_RESISTANCE", isBreakout(stocks, 10));
        features[featureIndex++] = normalizeScore("DONCHIAN", donchianBreakout(stocks, 5));
        features[featureIndex++] = normalizeScore("VOLATILITY_THRESHOLD", isVolatilitySpike(stocks, 10));
        features[featureIndex++] = normalizeScore("VOLATILITY_MONITOR", rollingVolatilityRatio(stocks, 5, 10, symbol));

        // Patterns Indicators
        features[featureIndex++] = normalizeScore("CONSECUTIVE_POSITIVE_CLOSES", consecutivePositiveCloses(stocks, 0.2));
        features[featureIndex++] = normalizeScore("HIGHER_HIGHS", isHigherHighs(stocks, 3));
        features[featureIndex++] = normalizeScore("FRACTAL_BREAKOUT", isFractalBreakout(stocks, 5, 10));
        features[featureIndex++] = normalizeScore("CANDLE_PATTERN", detectCandlePatterns(stocks.get(stocks.size() - 1), stocks.get(stocks.size() - 2)));
        features[featureIndex++] = normalizeScore("TRENDLINE", isTrendlineBreakout(stocks, 18));

        // Statistical Indicators
        features[featureIndex++] = normalizeScore("Z_SCORE", isZScoreSpike(stocks, 10, symbol));
        features[featureIndex++] = normalizeScore("CUMULATIVE_PERCENTAGE", isCumulativeSpike(stocks, 10, 5));
        features[featureIndex++] = normalizeScore("CUMULATIVE_THRESHOLD", cumulativePercentageChange(stocks, 5));

        // Advanced Indicators
        features[featureIndex++] = normalizeScore("BREAKOUT_MA", isBreakoutAboveMA(stocks, 10, true));
        features[featureIndex++] = normalizeScore("PARABOLIC", isParabolicSARBullish(stocks, 10, 0.2));
        features[featureIndex++] = normalizeScore("KELTNER", isKeltnerBreakout(stocks, 20, 10, 0.3));
        features[featureIndex++] = normalizeScore("ELDER_RAY", elderRayIndex(stocks, 20));
        features[featureIndex++] = normalizeScore("VOLUME_SPIKE", isVolumeSpike(stocks, 10, 0.03));
        features[featureIndex++] = normalizeScore("ATR", calculateATR(stocks, 15));


        float[] floatFeatures = new float[features.length];
        for (int i = 0; i < features.length; i++) {
            floatFeatures[i] = (float) features[i];
        }

        return floatFeatures;
    }

    /**
     * Generates notifications based on patterns and criteria within a frame of stock data.
     *
     * @param stocks The frame of stock data.
     * @param symbol The name of the stock being analyzed.
     * @return A list of notifications generated from the frame.
     */
    public static List<Notification> getNotificationForFrame(List<StockUnit> stocks, String symbol) {
        TimeSeries timeSeries = new TimeSeries(symbol);

        // Prevent notifications if time frame spans over the weekend (Friday to Monday)
        if (isWeekendSpan(stocks)) {
            return new ArrayList<>(); // Return empty list to skip notifications
        }

        Map<String, Double> aggregatedWeights = new HashMap<>();

        INDICATOR_RANGE_FULL.forEach((category, indicators) -> {
            double categoryWeight = INDICATOR_WEIGHTS_FULL.getOrDefault(category, 0.0);
            indicators.forEach((indicator, weight) -> {
                double finalWeight = weight * categoryWeight;
                aggregatedWeights.merge(indicator, finalWeight, Double::sum);
            });
        });

        float[] features = computeFeatures(stocks, symbol);
        double[] weightedFeatures = new double[features.length];

        // Map indicators to feature index
        Map<String, Integer> indicatorToIndex = new HashMap<>();
        int idx = 0;
        for (String indicator : INDICATOR_RANGE_MAP.keySet()) {
            indicatorToIndex.put(indicator, idx++);
        }

        // Apply weights
        aggregatedWeights.forEach((indicator, weight) -> {
            Integer index = indicatorToIndex.get(indicator);
            if (index != null && index < features.length) {
                weightedFeatures[index] = features[index] * weight;
            }
        });

        //feed normalized unweighted features
        double prediction = predict(features);

        for (StockUnit stockUnit : stocks) {
            timeSeries.add(new Minute(stockUnit.getDateDate()), stockUnit.getClose());
        }

        saveFeaturesToCSV(features, INDICATOR_RANGE_MAP, stocks.get(stocks.size() - 1).getDateDate(), prediction);
        return evaluateResult(timeSeries, weightedFeatures, prediction, aggregatedWeights, stocks, symbol);
    }

    // Method for evaluating results
    private static List<Notification> evaluateResult(TimeSeries timeSeries, double[] weightedFeatures, double prediction, Map<String, Double> aggregatedWeights, List<StockUnit> stocks, String symbol) {
        List<Notification> alertsList = new ArrayList<>();

        if (P_L_Tester.debug) {
            int i = 0;
            for (Map.Entry<String, Double> entry : aggregatedWeights.entrySet()) {
                if (i < weightedFeatures.length) {
                    System.out.println("Key: " + entry.getKey() + ", Weighted Feature: " + weightedFeatures[i]);
                }
                i++;
            }
        }

        createNotification(symbol, stocks.stream()
                .skip(stocks.size() - 4)
                .mapToDouble(StockUnit::getPercentageChange)
                .sum(), alertsList, timeSeries, stocks.get(stocks.size() - 1).getLocalDateTimeDate(), false, prediction);

        return alertsList;
    }

    //Indicators
    // 1. Simple Moving Average (SMA) Crossovers
    public static int isSMACrossover(List<StockUnit> window, int shortPeriod, int longPeriod) {
        if (window.size() < longPeriod) return 0;

        double shortSMA = window.stream()
                .skip(window.size() - shortPeriod)
                .mapToDouble(StockUnit::getClose)
                .average()
                .orElse(0);

        double longSMA = window.stream()
                .skip(window.size() - longPeriod)
                .mapToDouble(StockUnit::getClose)
                .average()
                .orElse(0);

        // Get previous values for crossover detection
        double prevShort = window.get(window.size() - shortPeriod - 1).getClose();
        double prevLong = window.get(window.size() - longPeriod - 1).getClose();

        // Return 1 if true, 0 if false
        return (shortSMA > longSMA && prevShort <= prevLong) ? 1 : 0;
    }

    // 2. EMA Crossover with Fast Calculation
    public static int isEMACrossover(List<StockUnit> window, int shortPeriod, int longPeriod) {
        if (window.size() < longPeriod) return 0;

        double shortEMA = calculateEMA(window, shortPeriod);
        double longEMA = calculateEMA(window, longPeriod);

        // Previous EMA values
        List<StockUnit> prevWindow = window.subList(0, window.size() - 1);
        double prevShort = calculateEMA(prevWindow, shortPeriod);
        double prevLong = calculateEMA(prevWindow, longPeriod);

        // Return 1 if true, 0 if false
        return (shortEMA > longEMA && prevShort <= prevLong) ? 1 : 0;
    }

    // 3. Price Crossing Key Moving Average
    public static int isPriceAboveSMA(List<StockUnit> window, int period) {
        if (window.size() < period) return 0;

        double sma = window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getClose)
                .average()
                .orElse(0);

        double currentPrice = window.get(window.size() - 1).getClose();
        double previousPrice = window.get(window.size() - 2).getClose();

        // Return 1 if true, 0 if false
        return (currentPrice > sma && previousPrice <= sma) ? 1 : 0;
    }

    // 4. MACD with Histogram
    public static Map<String, Double> calculateMACD(List<StockUnit> window) {
        final int SHORT_EMA = 12;
        final int LONG_EMA = 26;
        final int SIGNAL_EMA = 9;

        double macdLine = calculateEMA(window, SHORT_EMA) - calculateEMA(window, LONG_EMA);

        // Calculate Signal Line (EMA of MACD)
        List<Double> macdValues = window.stream()
                .map(su -> calculateEMA(Collections.singletonList(su), SHORT_EMA) - calculateEMA(Collections.singletonList(su), LONG_EMA))
                .collect(Collectors.toList());

        double signalLine = calculateEMAForValues(macdValues, SIGNAL_EMA);

        return Map.of(
                "macd", macdLine,
                "signal", signalLine,
                "histogram", macdLine - signalLine
        );
    }

    private static double calculateEMAForValues(List<Double> values, int period) {
        double smoothing = 2.0 / (period + 1);

        // Initial EMA is the first value in the list
        double ema = values.get(0);

        // Loop over the rest of the values and apply the EMA formula incrementally
        for (int i = 1; i < values.size(); i++) {
            ema = values.get(i) * smoothing + ema * (1 - smoothing);
        }

        return ema;
    }

    private static double calculateEMA(List<StockUnit> window, int period) {
        double smoothing = 2.0 / (period + 1);
        double ema = window.get(0).getClose();

        for (int i = 1; i < window.size(); i++) {
            ema = window.get(i).getClose() * smoothing + ema * (1 - smoothing);
        }
        return ema;
    }

    private static double calculateSMA(List<StockUnit> window, int period) {
        int start = Math.max(0, window.size() - period);
        double sum = 0;

        // Manual loop for better performance on small periods
        for (int i = start; i < window.size(); i++) {
            sum += window.get(i).getClose();
        }
        return sum / (window.size() - start);
    }

    // 5. TRIX Indicator
    public static double calculateTRIX(List<StockUnit> window, int period) {
        // Triple smoothing with EMAs
        List<Double> singleEMA = window.stream()
                .map(su -> calculateEMA(Collections.singletonList(su), period))
                .toList();

        List<Double> doubleEMA = singleEMA.stream()
                .map(val -> calculateEMAForValues(Collections.singletonList(val), period))
                .toList();

        List<Double> tripleEMA = doubleEMA.stream()
                .map(val -> calculateEMAForValues(Collections.singletonList(val), period))
                .toList();

        // Rate of Change
        double current = tripleEMA.get(tripleEMA.size() - 1);
        double previous = tripleEMA.get(tripleEMA.size() - 2);

        return ((current - previous) / previous) * 100;
    }

    // 6. Kaufman's Adaptive MA (KAMA)
    public static double calculateKAMA(List<StockUnit> window, int period) {
        List<Double> closes = window.parallelStream()
                .map(StockUnit::getClose)
                .collect(Collectors.toList());

        double efficiencyRatio = calculateEfficiencyRatio(closes, period);
        double fastSC = 2.0 / (2 + 1);
        double slowSC = 2.0 / (30 + 1);
        double smoothSC = efficiencyRatio * (fastSC - slowSC) + slowSC;

        double kama = closes.subList(0, period).stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0);

        for (int i = period; i < closes.size(); i++) {
            kama = kama + smoothSC * (closes.get(i) - kama);
        }

        return kama;
    }

    private static double calculateEfficiencyRatio(List<Double> prices, int period) {
        double direction = Math.abs(prices.get(prices.size() - 1) - prices.get(prices.size() - period));
        double volatility = IntStream.range(0, period)
                .mapToDouble(i -> Math.abs(prices.get(prices.size() - i - 1) - prices.get(prices.size() - i - 2)))
                .sum();

        return direction / volatility;
    }

    // 7. Relative Strength Index (RSI)
    public static double calculateRSI(List<StockUnit> window, int period) {
        if (window.size() < period + 1) return 0;

        List<Double> gains = new ArrayList<>();
        List<Double> losses = new ArrayList<>();

        // Parallel processing for price changes
        IntStream.range(1, period + 1).parallel().forEach(i -> {
            double change = window.get(i).getClose() - window.get(i - 1).getClose();
            if (change > 0) {
                synchronized (gains) {
                    gains.add(change);
                }
            } else {
                synchronized (losses) {
                    losses.add(Math.abs(change));
                }
            }
        });

        double avgGain = gains.parallelStream().mapToDouble(Double::doubleValue).average().orElse(0);
        double avgLoss = losses.parallelStream().mapToDouble(Double::doubleValue).average().orElse(0);

        if (avgLoss == 0) return 100;
        double rs = avgGain / avgLoss;
        return 100 - (100 / (1 + rs));
    }

    // 8. Rate of Change (ROC) with SIMD optimization
    public static double calculateROC(List<StockUnit> window, int periods) {
        if (window.size() < periods + 1) return 0;

        double[] closes = window.stream()
                .skip(window.size() - periods - 1)
                .mapToDouble(StockUnit::getClose)
                .toArray();

        // Manual vectorization for ARM NEON
        double current = closes[closes.length - 1];
        double past = closes[0];
        return ((current - past) / past) * 100;
    }

    // 9. Momentum Oscillator with look back optimization
    public static double calculateMomentum(List<StockUnit> window, int periods) {
        if (window.size() < periods + 1) return 0;

        double currentClose = window.get(window.size() - 1).getClose();
        double pastClose = window.get(window.size() - 1 - periods).getClose();
        return currentClose - pastClose;
    }

    // 10. Chande Momentum Oscillator (CMO) with branch less programming
    public static double calculateCMO(List<StockUnit> window, int period) {
        if (window.size() < period + 1) return 0;

        double sumUp = 0, sumDown = 0;
        for (int i = window.size() - period; i < window.size(); i++) {
            double change = window.get(i).getClose() - window.get(i - 1).getClose();
            sumUp += Math.max(change, 0);
            sumDown += Math.abs(Math.min(change, 0));
        }

        if (sumUp + sumDown == 0) return 0;
        return 100 * ((sumUp - sumDown) / (sumUp + sumDown));
    }

    // 11. Momentum Acceleration (2nd Derivative) with finite difference
    public static double calculateAcceleration(List<StockUnit> window, int period) {
        if (window.size() < 2 * period + 1) return 0;

        // Calculate momentum values over a sliding window
        double[] momentum = IntStream.range(0, window.size() - period)
                .mapToDouble(i -> calculateMomentum(window.subList(i, i + period + 1), period))
                .toArray();

        // Ensure sufficient data points for central difference
        int len = momentum.length;
        if (len < 3) return 0;

        // Compute second derivative using central difference
        return (momentum[len - 1] - 2 * momentum[len - 2] + momentum[len - 3]) / Math.pow(period, 2);
    }

    // 12. Bollinger Bands with Bandwidth Expansion
    public static Map<String, Double> calculateBollingerBands(List<StockUnit> window, int period) {
        if (window.size() < period) return Collections.emptyMap();

        DoubleSummaryStatistics stats = window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getClose)
                .summaryStatistics();

        double sma = stats.getAverage();
        double stdDev = Math.sqrt(window.stream()
                .skip(window.size() - period)
                .mapToDouble(su -> Math.pow(su.getClose() - sma, 2))
                .sum() / period);

        double upper = sma + 2 * stdDev;
        double lower = sma - 2 * stdDev;
        double bandwidth = (upper - lower) / sma;

        return Map.of(
                "upper", upper,
                "lower", lower,
                "bandwidth", bandwidth,
                "prev_bandwidth", getPreviousBandwidth(window, period) // Memoized
        );
    }

    private static double getPreviousBandwidth(List<StockUnit> window, int period) {
        if (window.size() < period + 1) return 0;
        return calculateBollingerBands(window.subList(0, window.size() - 1), period).get("bandwidth");
    }

    // 13. Breakout Above Resistance (Optimized with Sliding Window)
    public static int isBreakout(List<StockUnit> window, int resistancePeriod) {
        if (window.size() < resistancePeriod + 1) return 0;

        double currentClose = window.get(window.size() - 1).getClose();
        double resistance = window.stream()
                .skip(window.size() - resistancePeriod - 1)
                .limit(resistancePeriod)
                .mapToDouble(StockUnit::getClose)
                .max()
                .orElse(Double.MIN_VALUE);

        // Return 1 if true, 0 if false
        return (currentClose > resistance && window.get(window.size() - 2).getClose() <= resistance) ? 1 : 0;
    }

    // 14. Donchian Channel Breakout (Efficient Rolling Max)
    public static int donchianBreakout(List<StockUnit> window, int period) {
        if (window.size() < period + 1) return 0;

        double currentClose = window.get(window.size() - 1).getClose();

        // Find the maximum close in the last `period` bars
        double currentMax = window.subList(window.size() - period - 1, window.size() - 1).stream()
                .mapToDouble(StockUnit::getClose)
                .max()
                .orElse(Double.NEGATIVE_INFINITY);

        // Return 1 if breakout, 0 otherwise
        return (currentClose > currentMax && window.get(window.size() - 2).getClose() <= currentMax) ? 1 : 0;
    }

    // 15. Statistical Volatility Threshold (Reuse Bollinger stdDev)
    public static int isVolatilitySpike(List<StockUnit> window, int period) {
        if (window.size() < period) return 0;

        double currentChange = window.get(window.size() - 1).getPercentageChange();
        double mean = window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getPercentageChange)
                .average()
                .orElse(0);

        double stdDev = Math.sqrt(window.stream()
                .skip(window.size() - period)
                .mapToDouble(su -> Math.pow(su.getPercentageChange() - mean, 2))
                .average()
                .orElse(0));

        // Return 1 if true, 0 if false
        return Math.abs(currentChange) > 2 * stdDev ? 1 : 0;
    }

    // 16. Rolling Volatility Monitor (Incremental Calculation)
    public static double rollingVolatilityRatio(List<StockUnit> window, int shortPeriod, int longPeriod, String symbol) {
        DoubleArrayWindow vWindow = volatilityWindows.computeIfAbsent(symbol,
                k -> new DoubleArrayWindow(Math.max(shortPeriod, longPeriod) * 2));

        // Update with latest volatility (reuse Bollinger stdDev)
        double currentVolatility = calculateBollingerBands(window, 20).get("bandwidth");
        vWindow.add(currentVolatility);

        double shortTerm = vWindow.recentAverage(shortPeriod);
        double longTerm = vWindow.historicalAverage(longPeriod);

        return longTerm > 0 ? shortTerm / longTerm : 0;
    }

    // 17. Consecutive Positive Closes with Momentum Tolerance
    public static int consecutivePositiveCloses(List<StockUnit> window, double dipTolerance) {
        int count = 0;
        double prevClose = window.get(0).getClose();

        for (int i = 1; i < window.size(); i++) {
            double currentClose = window.get(i).getClose();
            double change = (currentClose - prevClose) / prevClose * 100;

            if (change > -dipTolerance) { // Allow small dips
                count = (change > 0) ? count + 1 : count;
            } else {
                count = 0;
            }
            prevClose = currentClose;
        }
        return count;
    }

    // 18. Higher Highs Pattern with Adaptive Window
    public static int isHigherHighs(List<StockUnit> window, int minConsecutive) {
        if (window.size() < minConsecutive) return 0;

        for (int i = window.size() - minConsecutive; i < window.size() - 1; i++) {
            if (window.get(i + 1).getClose() <= window.get(i).getClose()) {
                return 0;
            }
        }
        return 1;
    }

    // 19. Fractal Breakout Detection with Consolidation Phase
    public static int isFractalBreakout(List<StockUnit> window, int consolidationPeriod, double volatilityThreshold) {
        if (window.size() < consolidationPeriod + 2) return 0;

        // Calculate consolidation range
        double consolidationHigh = window.stream()
                .skip(window.size() - consolidationPeriod - 1)
                .limit(consolidationPeriod)
                .mapToDouble(StockUnit::getHigh)
                .max().orElse(0);

        double consolidationLow = window.stream()
                .skip(window.size() - consolidationPeriod - 1)
                .limit(consolidationPeriod)
                .mapToDouble(StockUnit::getLow)
                .min().orElse(0);

        double currentClose = window.get(window.size() - 1).getClose();
        double rangeSize = consolidationHigh - consolidationLow;

        // Return 1 if true, 0 if false
        return (currentClose > consolidationHigh && rangeSize / consolidationLow < volatilityThreshold) ? 1 : 0;
    }

    // 20. Candle Pattern Recognition (Optimized Bitmask Approach)
    public static int detectCandlePatterns(StockUnit current, StockUnit previous) {
        int patternMask = 0;

        // Hammer detection
        boolean isHammer = (current.getHigh() - current.getLow()) > 3 * (current.getClose() - current.getOpen()) &&
                (current.getClose() > current.getOpen()) &&
                (current.getClose() - current.getLow()) > 0.7 * (current.getHigh() - current.getLow());

        // Bullish Engulfing
        boolean isEngulfing = previous.getClose() < previous.getOpen() &&
                current.getClose() > previous.getOpen() &&
                current.getOpen() < previous.getClose();

        // Morning Star (simplified)
        boolean isMorningStar = previous.getClose() < previous.getOpen() &&
                current.getOpen() > previous.getClose() &&
                current.getClose() > previous.getOpen();

        if (isHammer) patternMask |= 0b1;
        if (isEngulfing) patternMask |= 0b10;
        if (isMorningStar) patternMask |= 0b100;

        return patternMask;
    }

    // 21. Automated Trend-line Analysis
    public static int isTrendlineBreakout(List<StockUnit> window, int lookBack) {
        if (window.size() < lookBack + 2) return 0;

        // Find pivot highs for the trend line
        List<Double> pivotHighs = new ArrayList<>();
        for (int i = 1; i < lookBack - 1; i++) {
            int idx = window.size() - 1 - i;
            if (idx <= 0 || idx >= window.size() - 1) continue;

            StockUnit p = window.get(idx);
            if (p.getHigh() > window.get(idx - 1).getHigh() &&
                    p.getHigh() > window.get(idx + 1).getHigh()) {
                pivotHighs.add(p.getHigh());
            }
        }

        if (pivotHighs.size() < 2) return 0;

        // Linear regression of pivot highs
        double expectedHigh = getExpectedHigh(pivotHighs);
        double currentClose = window.get(window.size() - 1).getClose();

        // Return 1 if breakout, 0 otherwise
        return (currentClose > expectedHigh && currentClose > window.get(window.size() - 2).getClose()) ? 1 : 0;
    }

    private static double getExpectedHigh(List<Double> pivotHighs) {
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for (int i = 0; i < pivotHighs.size(); i++) {
            sumX += i;
            sumY += pivotHighs.get(i);
            sumXY += i * pivotHighs.get(i);
            sumX2 += i * i;
        }

        double n = pivotHighs.size();
        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double intercept = (sumY - slope * sumX) / n;

        // Project trend line to current period
        return slope * (n + 1) + intercept;
    }

    // 22. Z-Score of Returns using incremental calculation
    public static int isZScoreSpike(List<StockUnit> window, int period, String symbol) {
        DoubleArrayWindow returnsWindow = returnsWindows.computeIfAbsent(symbol, k -> new DoubleArrayWindow(period * 2));

        // Update window with latest return
        double currentReturn = window.get(window.size() - 1).getPercentageChange();
        returnsWindow.add(currentReturn);

        // Get historical stats
        double mean = returnsWindow.historicalAverage(period);
        double stdDev = calculateStdDev(returnsWindow, period);

        // Return 1 if true, 0 if false
        return (stdDev != 0 && (currentReturn - mean) / stdDev >= 2.0) ? 1 : 0;
    }

    private static double calculateStdDev(DoubleArrayWindow window, int period) {
        double mean = window.historicalAverage(period);
        double sumSq = 0;
        int count = Math.min(period, window.filled ? window.values.length : window.index);

        for (int i = 0; i < count; i++) {
            int pos = (window.index - count + i) % window.values.length;
            sumSq += Math.pow(window.values[pos] - mean, 2);
        }
        return Math.sqrt(sumSq / count);
    }

    // 23. Cumulative Percentage Change with threshold check
    public static int isCumulativeSpike(List<StockUnit> window, int period, double threshold) {
        if (window.size() < period) return 0;

        double sum = window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getPercentageChange)
                .sum();

        // Return 1 if true, 0 if false
        return sum >= threshold ? 1 : 0;
    }

    // 24. Cumulative Percentage Change
    private static double cumulativePercentageChange(List<StockUnit> stocks, int lastChangeLength) {
        int startIndex = 0;
        try {
            startIndex = stocks.size() - lastChangeLength;
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Using parallelStream to process the list in parallel
        return stocks.subList(startIndex, stocks.size())
                .stream()   // Parallel processing
                .mapToDouble(StockUnit::getPercentageChange)  // Convert to double
                .sum();  // Sum all the results
    }

    // 25. Breakout Above Moving Average
    public static int isBreakoutAboveMA(List<StockUnit> window, int period, boolean useEMA) {
        // Validation
        if (window == null || window.size() < period || period <= 0) {
            return 0;  // Return 0 if invalid
        }

        // Get current and previous prices
        double currentClose = window.get(window.size() - 1).getClose();
        double previousClose = window.size() >= 2 ?
                window.get(window.size() - 2).getClose() : currentClose;

        // Calculate MAs with caching
        double currentMA = calculateMA(window, period, useEMA);
        double previousMA = calculateMA(window.subList(0, window.size() - 1), period, useEMA);

        // Return 1 if true, 0 if false
        return (currentClose > currentMA && previousClose <= previousMA) ? 1 : 0;
    }

    private static double calculateMA(List<StockUnit> window, int period, boolean useEMA) {
        return useEMA ? calculateEMA(window, period) : calculateSMA(window, period);
    }

    // 26. Parabolic SAR Approximation using Close Prices
    public static int isParabolicSARBullish(List<StockUnit> window, int period, double acceleration) {
        if (window.size() < period + 2) return 0;

        double[] sarValues = new double[window.size()];
        sarValues[0] = window.get(0).getClose();
        boolean uptrend = true;
        double extremePoint = window.get(0).getClose();

        for (int i = 1; i < window.size(); i++) {
            double close = window.get(i).getClose();
            double prevSAR = sarValues[i - 1];

            if (uptrend) {
                sarValues[i] = prevSAR + acceleration * (extremePoint - prevSAR);
                if (close < sarValues[i]) {
                    uptrend = false;
                    sarValues[i] = extremePoint;
                    extremePoint = close;
                } else {
                    extremePoint = Math.max(extremePoint, close);
                }
            } else {
                sarValues[i] = prevSAR - acceleration * (prevSAR - extremePoint);
                if (close > sarValues[i]) {
                    uptrend = true;
                    sarValues[i] = extremePoint;
                    extremePoint = close;
                } else {
                    extremePoint = Math.min(extremePoint, close);
                }
            }
        }

        StockUnit current = window.get(window.size() - 1);
        // Return 1 if bullish, 0 if not
        return (current.getClose() > sarValues[sarValues.length - 1]) ? 1 : 0;
    }

    // 27. Keltner Channels Breakout
    public static int isKeltnerBreakout(List<StockUnit> window, int emaPeriod, int atrPeriod, double multiplier) {
        double ema = calculateEMA(window, emaPeriod);
        double atr = calculateATR(window, atrPeriod);
        double upperBand = ema + (multiplier * atr);

        // Return 1 if the close is greater than the upper band, otherwise return 0
        return (window.get(window.size() - 1).getClose() > upperBand) ? 1 : 0;
    }

    // 28. Elder-Ray Index Approximation
    public static double elderRayIndex(List<StockUnit> window, int emaPeriod) {
        double ema = calculateEMA(window, emaPeriod);
        return window.get(window.size() - 1).getClose() - ema;
    }

    // 29. Volume Spike Detection with Adaptive Threshold
    public static int isVolumeSpike(List<StockUnit> window, int period, double thresholdFactor) {
        if (window.size() < period) return 0;

        // Calculate average volume
        double avgVolume = calculateAverageVolume(window, period);

        // Get the current volume (last element in the window)
        double currentVolume = window.get(window.size() - 1).getVolume();

        // Return 1 if true, 0 if false
        return currentVolume > (avgVolume * thresholdFactor) ? 1 : 0;
    }

    private static double calculateAverageVolume(List<StockUnit> window, int period) {
        return window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getVolume)
                .average()
                .orElse(0);
    }

    // 30. ATR Calculator using Close Prices (adjusted from traditional)
    private static double calculateATR(List<StockUnit> window, int period) {
        double atrSum = 0;
        for (int i = 1; i < window.size(); i++) {
            double high = window.get(i).getHigh();
            double low = window.get(i).getLow();
            double prevClose = window.get(i - 1).getClose();

            double trueRange = Math.max(high - low, Math.max(Math.abs(high - prevClose), Math.abs(low - prevClose)));
            atrSum += trueRange;
        }

        return atrSum / period;
    }

    /**
     * Checks if the time frame of stock data spans over a weekend or across days.
     *
     * @param stocks The frame of stock data.
     * @return True if the time frame spans a weekend or multiple days; false otherwise.
     */
    private static boolean isWeekendSpan(List<StockUnit> stocks) {
        // Validate list to avoid errors
        if (stocks == null || stocks.isEmpty()) {
            throw new IllegalArgumentException("Stock list cannot be null or empty");
        }

        // Parse dates using the provided method
        LocalDateTime startDate = stocks.get(0).getLocalDateTimeDate();
        LocalDateTime endDate = stocks.get(stocks.size() - 1).getLocalDateTimeDate();

        // Check if the time span includes a weekend or spans different days
        return (startDate.getDayOfWeek() == DayOfWeek.FRIDAY && endDate.getDayOfWeek() == DayOfWeek.MONDAY) || !startDate.toLocalDate().equals(endDate.toLocalDate());
    }

    /**
     * Creates a notification for a stock event (increase or dip) and adds it to the alerts list.
     *
     * @param symbol      The name of the stock.
     * @param totalChange The total percentage change triggering the notification.
     * @param alertsList  The list to store the notification.
     * @param timeSeries  The time series for graphical representation.
     * @param date        The date of the event.
     * @param dip         True if the event is a dip; false if it's an increase.
     */
    private static void createNotification(String symbol, double totalChange, List<Notification> alertsList, TimeSeries timeSeries, LocalDateTime date, boolean dip, double prediction) {
        if ((totalChange > 0) && !dip) {
            alertsList.add(new Notification(String.format("%.3f%% %s ↑ %s", totalChange, symbol, prediction), String.format("Increased by %.3f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))), timeSeries, new Color(50, 205, 50), date, symbol, totalChange));
        } else if (dip) {
            alertsList.add(new Notification(String.format("%.3f%% %s ↓ %s", totalChange, symbol, prediction), String.format("dipped by %.3f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))), timeSeries, new Color(255, 217, 0), date, symbol, totalChange));
        }
    }

    public static List<StockUnit> getSymbolTimeline(String symbol) {
        return Collections.unmodifiableList(
                symbolTimelines.getOrDefault(symbol.toUpperCase(), new ArrayList<>())
        );
    }

    //Interfaces
    public interface DataCallback {
        void onDataFetched(Double[] values);
    }

    public interface TimelineCallback {
        void onTimeLineFetched(List<StockUnit> stocks);
    }

    public interface SymbolSearchCallback {
        void onSuccess(List<String> symbols);

        void onFailure(Exception e);
    }

    public interface ReceiveNewsCallback {
        void onNewsReceived(List<NewsResponse.NewsItem> news);
    }

    public interface SymbolCallback {
        void onSymbolsAvailable(List<String> symbols);
    }

    public interface RealTimeCallback {
        void onRealTimeReceived(RealTimeResponse.RealTimeMatch value);
    }

    // Helper class for efficient rolling statistics
    private static class DoubleArrayWindow {
        private final double[] values;
        private int index = 0;
        private boolean filled = false;

        public DoubleArrayWindow(int size) {
            this.values = new double[size];
        }

        public void add(double value) {
            values[index++ % values.length] = value;
            if (index >= values.length) filled = true;
        }

        public double recentAverage(int period) {
            int count = Math.min(period, filled ? values.length : index);
            return Arrays.stream(values, 0, count).parallel().average().orElse(0);
        }

        public double historicalAverage(int period) {
            int start = Math.max(0, (filled ? values.length : index) - period);
            int count = Math.min(period, filled ? values.length : index - start);
            return Arrays.stream(values, start, start + count).parallel().average().orElse(0);
        }
    }
}