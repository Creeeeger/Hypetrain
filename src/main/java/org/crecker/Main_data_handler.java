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

public class Main_data_handler {
    private static final Map<String, Deque<Double>> donchianCache = new ConcurrentHashMap<>();
    private static final Map<String, DoubleArrayWindow> volatilityWindows = new ConcurrentHashMap<>();
    private static final Map<String, DoubleArrayWindow> returnsWindows = new ConcurrentHashMap<>();

    private static final Map<String, Map<String, Double>> INDICATOR_RANGE_MAP = Map.ofEntries(
            // Trend Following Indicators
            Map.entry("SMA_CROSS", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("EMA_CROSS", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("PRICE_SMA_DISTANCE", Map.of("min", -20.0, "max", 20.0)),
            Map.entry("MACD", Map.of("min", -5.0, "max", 5.0)),
            Map.entry("TRIX", Map.of("min", -5.0, "max", 5.0)),
            Map.entry("KAMA", Map.of("min", -20.0, "max", 20.0)),

            // Momentum Indicators
            Map.entry("RSI", Map.of("min", 0.0, "max", 100.0)),
            Map.entry("ROC", Map.of("min", -100.0, "max", 100.0)),
            Map.entry("MOMENTUM", Map.of("min", -100.0, "max", 100.0)),
            Map.entry("CMO", Map.of("min", -100.0, "max", 100.0)),
            Map.entry("ACCELERATION", Map.of("min", -10.0, "max", 10.0)),

            // Volatility & Breakouts Indicators
            Map.entry("BOLLINGER", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("BREAKOUT_RESISTANCE", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("DONCHIAN", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("VOLATILITY_THRESHOLD", Map.of("min", -3.0, "max", 3.0)),
            Map.entry("VOLATILITY_MONITOR", Map.of("min", 0.0, "max", 5.0)),

            // Patterns Indicators
            Map.entry("CONSECUTIVE_POSITIVE_CLOSES", Map.of("min", 0.0, "max", 10.0)),
            Map.entry("HIGHER_HIGHS", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("FRACTAL_BREAKOUT", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("CANDLE_PATTERN", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("TRENDLINE", Map.of("min", 0.0, "max", 1.0)),

            // Statistical Indicators
            Map.entry("Z_SCORE", Map.of("min", -3.0, "max", 3.0)),
            Map.entry("CUMULATIVE_PERCENTAGE", Map.of("min", -20.0, "max", 20.0)),
            Map.entry("CUMULATIVE_THRESHOLD", Map.of("min", 0.0, "max", 1.0)),

            // Advanced Indicators
            Map.entry("BREAKOUT_MA", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("PARABOLIC", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("KELTNER", Map.of("min", 0.0, "max", 1.0)),
            Map.entry("ELDER_RAY", Map.of("min", -10.0, "max", 10.0)),
            Map.entry("VOLUME_SPIKE", Map.of("min", 0.0, "max", 5.0)),
            Map.entry("ATR", Map.of("min", 0.0, "max", 1.0))
    );

    // Trend Following Indicators
    private static final Map<String, Double> TREND_FOLLOWING_WEIGHTS = Map.ofEntries(
            Map.entry("SMA_CROSS", 0.15),
            Map.entry("EMA_CROSS", 0.15),
            Map.entry("PRICE_SMA_DISTANCE", 0.20),
            Map.entry("MACD", 0.15),
            Map.entry("TRIX", 0.20),
            Map.entry("KAMA", 0.15)
    );

    // Momentum Indicators
    private static final Map<String, Double> MOMENTUM_WEIGHTS = Map.ofEntries(
            Map.entry("RSI", 0.25),
            Map.entry("ROC", 0.15),
            Map.entry("MOMENTUM", 0.15),
            Map.entry("CMO", 0.15),
            Map.entry("ACCELERATION", 0.30)
    );

    // Volatility & Breakouts Indicators
    private static final Map<String, Double> VOLATILITY_BREAKOUTS_WEIGHTS = Map.ofEntries(
            Map.entry("BOLLINGER", 0.20),
            Map.entry("BREAKOUT_RESISTANCE", 0.15),
            Map.entry("DONCHIAN", 0.20),
            Map.entry("VOLATILITY_THRESHOLD", 0.20),
            Map.entry("VOLATILITY_MONITOR", 0.25)
    );

    // Patterns Indicators
    private static final Map<String, Double> PATTERNS_WEIGHTS = Map.ofEntries(
            Map.entry("CONSECUTIVE_POSITIVE_CLOSES", 0.25),
            Map.entry("HIGHER_HIGHS", 0.15),
            Map.entry("FRACTAL_BREAKOUT", 0.20),
            Map.entry("CANDLE_PATTERN", 0.15),
            Map.entry("TRENDLINE", 0.25)
    );

    // Statistical Indicators
    private static final Map<String, Double> STATISTICAL_WEIGHTS = Map.ofEntries(
            Map.entry("Z_SCORE", 0.40),
            Map.entry("CUMULATIVE_PERCENTAGE", 0.40),
            Map.entry("CUMULATIVE_THRESHOLD", 0.20)
    );

    // Advanced Indicators
    private static final Map<String, Double> ADVANCED_WEIGHTS = Map.ofEntries(
            Map.entry("BREAKOUT_MA", 0.10),
            Map.entry("PARABOLIC", 0.10),
            Map.entry("KELTNER", 0.20),
            Map.entry("ELDER_RAY", 0.20),
            Map.entry("VOLUME_SPIKE", 0.25),
            Map.entry("ATR", 0.15)
    );

    // Aggregated weights map
    private static final Map<String, Map<String, Double>> INDICATOR_RANGE_FULL = Map.of(
            "TrendFollowing", TREND_FOLLOWING_WEIGHTS,
            "Momentum", MOMENTUM_WEIGHTS,
            "VolatilityBreakouts", VOLATILITY_BREAKOUTS_WEIGHTS,
            "Patterns", PATTERNS_WEIGHTS,
            "Statistical", STATISTICAL_WEIGHTS,
            "Advanced", ADVANCED_WEIGHTS
    );

    // Category Level Weights
    private static final Map<String, Double> INDICATOR_WEIGHTS_FULL = Map.of(
            "TrendFollowing", 0.20,
            "Momentum", 0.15,
            "VolatilityBreakouts", 0.25,
            "Patterns", 0.15,
            "Statistical", 0.10,
            "Advanced", 0.15
    );

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

//    // Normalize to [0,1] range based on indicator type
//    private static double normalizeScore(String indicator, double rawValue) {
//        Map<String, Double> range = INDICATOR_RANGE_MAP.get(indicator);
//        if (range == null) return rawValue; // No normalization
//
//        double min = range.get("min");
//        double max = range.get("max");
//
//        // Special handling for centered indicators
//        if ("MACD".equals(indicator)) {
//            return 0.5 + (rawValue / (max - min)); // Center at 0.5
//        }
//
//        return (rawValue - min) / (max - min);
//    }

    /**
     * Generates notifications based on patterns and criteria within a frame of stock data.
     *
     * @param stocks The frame of stock data.
     * @param symbol The name of the stock being analyzed.
     * @return A list of notifications generated from the frame.
     */
    public static List<Notification> getNotificationForFrame(List<StockUnit> stocks, String symbol) {
        //prevent wrong dip variables
        int lastChangeLength = 5;

        //minor dip detection variables
        int consecutiveIncreaseCount = 0;
        double cumulativeIncrease = 0;
        double cumulativeDecrease = 0;
        double minorDipTolerance = 0.65; // in %

        //rapid increase variables
        double minIncrease = 2; // in %
        int rapidWindowSize = 1;
        int minConsecutiveCount = 2;

        //Crash variables
        double dipDown = -1.5; //in %
        double dipUp = 0.8; //in %

        //algorithm related variables
        List<Notification> alertsList = new ArrayList<>();
        TimeSeries timeSeries = new TimeSeries(symbol);

        // Prevent notifications if time frame spans over the weekend (Friday to Monday)
        if (isWeekendSpan(stocks)) {
            return new ArrayList<>(); // Return empty list to skip notifications
        }

//        Map<String, Double> aggregatedWeights = new HashMap<>();
//        // Step 1: Compute weighted values per category
//        INDICATOR_RANGE_FULL.forEach((category, indicators) -> {
//            double categoryWeight = INDICATOR_WEIGHTS_FULL.getOrDefault(category, 0.0);
//            indicators.forEach((indicator, weight) -> {
//                double finalWeight = weight * categoryWeight;
//                aggregatedWeights.merge(indicator, finalWeight, Double::sum);
//            });
//        });
//
//        // Step 2: Display result weights
//        System.out.println("Final Indicator Weights:");
//        aggregatedWeights.entrySet().stream()
//                .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
//                .forEach(entry -> System.out.printf("%s: %.4f%n", entry.getKey(), entry.getValue()));


        for (int i = 1; i < stocks.size(); i++) {
            //Changes & percentages calculations
            double percentageChange = stocks.get(i).getPercentageChange();
            timeSeries.add(new Minute(stocks.get(i).getDateDate()), stocks.get(i).getClose());

            // Check if the current percentage change is positive or a minor dip (momentum calculation)
            if (percentageChange > 0) {
                cumulativeIncrease += percentageChange;
                consecutiveIncreaseCount++;
                cumulativeDecrease = 0; // Reset cumulative decrease when there's an increase
            } else {
                cumulativeDecrease += Math.abs(percentageChange); // Track cumulative decreases
                // Check if the cumulative decrease is within tolerance
                if (cumulativeDecrease <= minorDipTolerance * cumulativeIncrease) {
                    // Allow minor dip, continue momentum tracking without resetting
                    consecutiveIncreaseCount++;
                } else {
                    // If the dip is too large, reset momentum
                    consecutiveIncreaseCount = 0;
                    cumulativeIncrease = 0;
                    cumulativeDecrease = 0;
                }
            }

            //pattern detection & volatility calculation of percentage changes logic
            if (i == stocks.size() - 1) {
                double lastChanges = cumulativePercentageChange(stocks, lastChangeLength);
                boolean volTest = isVolatilitySpike(stocks, 5);

                if (((stocks.get(i - 1).getPercentageChange() +
                        stocks.get(i - 2).getPercentageChange() +
                        stocks.get(i - 3).getPercentageChange()) <= dipDown) &&
                        (stocks.get(i).getPercentageChange() >= dipUp)) {
                    try {
                        createNotification(symbol, lastChanges, alertsList, timeSeries, stocks.get(i).getLocalDateTimeDate(), true);
                    } catch (Exception ignored) {
                    }
                }

                //rapid increase logic
                rapidIncreaseLogic(stocks, symbol, i, rapidWindowSize,
                        minIncrease, consecutiveIncreaseCount,
                        minConsecutiveCount, lastChangeLength, lastChanges,
                        alertsList, timeSeries, volTest);
            }
        }
        return alertsList;
    }

    private static void rapidIncreaseLogic(List<StockUnit> stocks, String stockName, int i,
                                           int rapidWindowSize,
                                           double minIncrease, int consecutiveIncreaseCount, int minConsecutiveCount,
                                           int lastChangeLength, double lastChanges, List<Notification> alertsList,
                                           TimeSeries timeSeries, boolean volTest) {
        if (i >= rapidWindowSize) { // Ensure the window is valid
            if (volTest && (consecutiveIncreaseCount >= minConsecutiveCount) &&
                    (i >= (stocks.size() - lastChangeLength)) &&
                    (lastChanges > minIncrease)) {

                createNotification(stockName, lastChanges, alertsList, timeSeries, stocks.get(i).getLocalDateTimeDate(), false);

//                System.out.printf("Name: %s Consecutive %s vs %s, Last Change %.2f vs %.2f Date %s%n", stockName, consecutiveIncreaseCount, minConsecutiveCount, lastChanges, minIncrease, stocks.get(i).getDateDate());
            }
        }
    }

    //Indicators
    // 1. Simple Moving Average (SMA) Crossovers
    public static boolean isSMACrossover(List<StockUnit> window, int shortPeriod, int longPeriod) {
        if (window.size() < longPeriod) return false;

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

        return shortSMA > longSMA && prevShort <= prevLong;
    }

    // 2. EMA Crossover with Fast Calculation
    public static boolean isEMACrossover(List<StockUnit> window, int shortPeriod, int longPeriod) {
        if (window.size() < longPeriod) return false;

        double shortEMA = calculateEMA(window, shortPeriod);
        double longEMA = calculateEMA(window, longPeriod);

        // Previous EMA values
        List<StockUnit> prevWindow = window.subList(0, window.size() - 1);
        double prevShort = calculateEMA(prevWindow, shortPeriod);
        double prevLong = calculateEMA(prevWindow, longPeriod);

        return shortEMA > longEMA && prevShort <= prevLong;
    }

    // 3. Price Crossing Key Moving Average
    public static boolean isPriceAboveSMA(List<StockUnit> window, int period) {
        if (window.size() < period) return false;

        double sma = window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getClose)
                .average()
                .orElse(0);

        double currentPrice = window.get(window.size() - 1).getClose();
        double previousPrice = window.get(window.size() - 2).getClose();

        return currentPrice > sma && previousPrice <= sma;
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
        if (window.size() < period + 2) return 0;

        double[] momentum = IntStream.range(0, period + 1)
                .mapToDouble(i -> calculateMomentum(window.subList(i, window.size()), period))
                .toArray();

        // Second derivative using central difference
        return (momentum[momentum.length - 1] - 2 * momentum[momentum.length - 2]
                + momentum[momentum.length - 3]) / Math.pow(period, 2);
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
    public static boolean isBreakout(List<StockUnit> window, int resistancePeriod) {
        if (window.size() < resistancePeriod + 1) return false;

        double currentClose = window.get(window.size() - 1).getClose();
        double resistance = window.stream()
                .skip(window.size() - resistancePeriod - 1)
                .limit(resistancePeriod)
                .mapToDouble(StockUnit::getClose)
                .max()
                .orElse(Double.MIN_VALUE);

        return currentClose > resistance &&
                window.get(window.size() - 2).getClose() <= resistance;
    }

    // 14. Donchian Channel Breakout (Efficient Rolling Max)
    public static boolean donchianBreakout(List<StockUnit> window, int period, String symbol) {
        Deque<Double> maxQueue = donchianCache.computeIfAbsent(symbol, k -> new ArrayDeque<>());
        double currentClose = window.get(window.size() - 1).getClose();

        // Maintain rolling window of highs
        if (maxQueue.size() >= period) {
            maxQueue.pollFirst();
        }
        maxQueue.addLast(currentClose);

        double currentMax = new ArrayList<>(maxQueue)
                .stream()
                .filter(Objects::nonNull)  // Filter out null values
                .max(Double::compare)
                .orElse(0.0);

        return currentClose > currentMax && window.get(window.size() - 2).getClose() <= currentMax;
    }

    // 15. Statistical Volatility Threshold (Reuse Bollinger stdDev)
    public static boolean isVolatilitySpike(List<StockUnit> window, int period) {
        if (window.size() < period) return false;

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

        return Math.abs(currentChange) > 2 * stdDev;
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
    public static boolean isHigherHighs(List<StockUnit> window, int minConsecutive) {
        if (window.size() < minConsecutive) return false;

        for (int i = window.size() - minConsecutive; i < window.size() - 1; i++) {
            if (window.get(i + 1).getClose() <= window.get(i).getClose()) {
                return false;
            }
        }
        return true;
    }

    // 19. Fractal Breakout Detection with Consolidation Phase
    public static boolean isFractalBreakout(List<StockUnit> window, int consolidationPeriod, double volatilityThreshold) {
        if (window.size() < consolidationPeriod + 2) return false;

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

        // Breakout conditions
        return currentClose > consolidationHigh && rangeSize / consolidationLow < volatilityThreshold;
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
    public static boolean isTrendlineBreakout(List<StockUnit> window, int lookback) {
        if (window.size() < lookback + 2) return false;

        // Find pivot highs for trend line
        List<Double> pivotHighs = new ArrayList<>();
        for (int i = 3; i < lookback; i++) {
            StockUnit p = window.get(window.size() - i);
            if (p.getHigh() > window.get(window.size() - i - 1).getHigh() &&
                    p.getHigh() > window.get(window.size() - i + 1).getHigh()) {
                pivotHighs.add(p.getHigh());
            }
        }

        if (pivotHighs.size() < 2) return false;

        // Linear regression of pivot highs
        double expectedHigh = getExpectedHigh(pivotHighs);
        double currentClose = window.get(window.size() - 1).getClose();

        return currentClose > expectedHigh &&
                currentClose > window.get(window.size() - 2).getClose();
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
    public static boolean isZScoreSpike(List<StockUnit> window, int period, String symbol) {
        DoubleArrayWindow returnsWindow = returnsWindows.computeIfAbsent(symbol,
                k -> new DoubleArrayWindow(period * 2));

        // Update window with latest return
        double currentReturn = window.get(window.size() - 1).getPercentageChange();
        returnsWindow.add(currentReturn);

        // Get historical stats
        double mean = returnsWindow.historicalAverage(period);
        double stdDev = calculateStdDev(returnsWindow, period);

        return stdDev != 0 && (currentReturn - mean) / stdDev >= 2.0;
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
    public static boolean isCumulativeSpike(List<StockUnit> window, int period, double threshold) {
        if (window.size() < period) return false;

        double sum = window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getPercentageChange)
                .sum();

        return sum >= threshold;
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
    public static boolean isBreakoutAboveMA(List<StockUnit> window, int period, boolean useEMA) {
        // Validation
        if (window == null || window.size() < period || period <= 0) {
            return false;
        }

        // Get current and previous prices
        double currentClose = window.get(window.size() - 1).getClose();
        double previousClose = window.size() >= 2 ?
                window.get(window.size() - 2).getClose() : currentClose;

        // Calculate MAs with caching
        double currentMA = calculateMA(window, period, useEMA);
        double previousMA = calculateMA(window.subList(0, window.size() - 1), period, useEMA);

        return currentClose > currentMA && previousClose <= previousMA;
    }

    private static double calculateMA(List<StockUnit> window, int period, boolean useEMA) {
        return useEMA ? calculateEMA(window, period) : calculateSMA(window, period);
    }

    // 26. Parabolic SAR Approximation using Close Prices
    public static boolean isParabolicSARBullish(List<StockUnit> window, int period, double acceleration) {
        if (window.size() < period + 2) return false;

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
        return current.getClose() > sarValues[sarValues.length - 1];
    }

    // 27. Keltner Channels Breakout
    public static boolean isKeltnerBreakout(List<StockUnit> window, int emaPeriod, int atrPeriod, double multiplier) {
        double ema = calculateEMA(window, emaPeriod);
        double atr = calculateATR(window, atrPeriod);
        double upperBand = ema + (multiplier * atr);
        return window.get(window.size() - 1).getClose() > upperBand;
    }

    // 28. Elder-Ray Index Approximation
    public static double elderRayIndex(List<StockUnit> window, int emaPeriod) {
        double ema = calculateEMA(window, emaPeriod);
        return window.get(window.size() - 1).getClose() - ema;
    }

    // 29. Volume Spike Detection with Adaptive Threshold
    public static boolean isVolumeSpike(List<StockUnit> window, int period, double thresholdFactor) {
        if (window.size() < period) return false;

        // Calculate average volume
        double avgVolume = calculateAverageVolume(window, period);

        // Get the current volume (last element in the window)
        double currentVolume = window.get(window.size() - 1).getVolume();

        // Return true if the current volume is greater than the average volume times the threshold factor
        return currentVolume > (avgVolume * thresholdFactor);
    }

    private static double calculateAverageVolume(List<StockUnit> window, int period) {
        double totalVolume = 0;

        // Iterate over the window to sum the volume over the period
        for (int i = window.size() - period; i < window.size(); i++) {
            totalVolume += window.get(i).getVolume();
        }

        // Calculate and return the average volume
        return totalVolume / period;
    }

    // 30. ATR Calculator using Close Prices (adjusted from traditional)
    private static double calculateATR(List<StockUnit> window, int period) {
        double atr = Math.abs(window.get(1).getClose() - window.get(0).getClose());

        for (int i = 2; i < window.size(); i++) {
            atr += Math.abs(window.get(i).getClose() - window.get(i - 1).getClose());
        }

        return atr / period;
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
    private static void createNotification(String symbol, double totalChange, List<Notification> alertsList, TimeSeries timeSeries, LocalDateTime date, boolean dip) {
        String dateString = date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"));
        String dateStringShort = date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm"));

        if ((totalChange > 0) && !dip) {
            alertsList.add(new Notification(String.format("%.3f%% %s ↑ %s", totalChange, symbol, dateStringShort), String.format("Increased by %.3f%% at the %s", totalChange, dateString), timeSeries, new Color(50, 205, 50), date, symbol, totalChange));
        } else if (dip) {
            alertsList.add(new Notification(String.format("%.3f%% %s ↓ %s", totalChange, symbol, dateStringShort), String.format("dipped by %.3f%% at the %s", totalChange, dateString), timeSeries, new Color(255, 217, 0), date, symbol, totalChange));
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