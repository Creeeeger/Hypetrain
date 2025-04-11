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
import org.jfree.data.time.Second;
import org.jfree.data.time.TimeSeries;

import java.io.*;
import java.time.DayOfWeek;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import static org.crecker.RallyPredictor.predict;
import static org.crecker.mainUI.*;
import static org.crecker.pLTester.PLAnalysis;
import static org.crecker.pLTester.SYMBOLS;

public class mainDataHandler {
    private static final Map<String, Map<String, Map<String, Double>>> SYMBOL_INDICATOR_RANGES = new ConcurrentHashMap<>();

    public static final List<String> INDICATOR_KEYS = List.of(
            "SMA_CROSS",              // 0
            "TRIX",                   // 1
            "ROC",                    // 2
            "PLACEHOLDER",            // 3
            "CUMULATIVE_PERCENTAGE",  // 4
            "CUMULATIVE_THRESHOLD",   // 5
            "KELTNER",                // 6
            "ELDER_RAY"               // 7
    );

    public static final Set<String> BINARY_INDICATORS = Set.of(
            "KELTNER",
            "CUMULATIVE_PERCENTAGE"
    );

    private static final Map<String, Double> CATEGORY_WEIGHTS = new HashMap<>() {{
        /*
          Category	Bull 	Bear 	High Volatility Scraper
          TREND	    0.30	0.15	0.20            0.1
          MOMENTUM	0.40	0.25	0.35            0.1
          STATS     0.15    0.30	0.25            0.45
          ADVANCED	0.15	0.30	0.20            0.35
         */
        put("TREND", 0.1);    // Features 0-1 (SMA, TRIX)
        put("MOMENTUM", 0.1); // Feature 2 (ROC)
        put("STATISTICAL", 0.45);// Features 4-5 (Spike, Cumulative)
        put("ADVANCED", 0.35);  // Features 6-7 (Keltner, Elder)
    }};

    private static final Map<Integer, String> FEATURE_CATEGORIES = new HashMap<>() {{
        put(0, "TREND");
        put(1, "TREND");
        put(2, "MOMENTUM");
        put(3, "NEUTRAL");
        put(4, "STATISTICAL");
        put(5, "STATISTICAL");
        put(6, "ADVANCED");
        put(7, "ADVANCED");
    }};

    static final Map<String, List<StockUnit>> symbolTimelines = new ConcurrentHashMap<>();
    static final List<Notification> notificationsForPLAnalysis = new ArrayList<>();
    static final TimeSeries predictionTimeSeries = new TimeSeries("Predictions");
    private static final ConcurrentHashMap<String, Integer> smaStateMap = new ConcurrentHashMap<>();
    private static final ScheduledExecutorService executorService = Executors.newSingleThreadScheduledExecutor();
    private static final Map<String, Double> historicalATRCache = new ConcurrentHashMap<>();
    private static final int HISTORICAL_LOOK_BACK = 100;
    static int frameSize = 30; // Frame size for analysis
    public static final TimeSeries[] featureTimeSeriesArray = new TimeSeries[8];
    public static String[] stockSymbols = {
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

    static {
        // Initialize feature TimeSeries, excluding index 3
        for (int i = 0; i < featureTimeSeriesArray.length; i++) {
            if (i != 3) {
                featureTimeSeriesArray[i] = new TimeSeries("Feature " + i);
            }
        }
    }

    public static void main(String[] args) {
        PLAnalysis();
        //realTimeDataCollector("MARA");
    }

    public static void InitAPi(String token) {
        // Configure the API client
        Config cfg = Config.builder()
                .key(token)
                .timeOut(10) // Timeout in seconds
                .build();

        // Initialize the Alpha Vantage API
        AlphaVantage.api().init(cfg);
    }

    public static void getTimeline(String symbolName, TimelineCallback callback) {
        List<StockUnit> stocks = new ArrayList<>(); // Directly use a List<StockUnit>

        AlphaVantage.api()
                .timeSeries()
                .intraday()
                .forSymbol(symbolName)
                .interval(Interval.ONE_MIN)
                .outputSize(OutputSize.FULL)
                .onSuccess(e -> {
                    TimeSeriesResponse response = (TimeSeriesResponse) e;
                    stocks.addAll(response.getStockUnits()); // Populate the list
                    callback.onTimeLineFetched(stocks); // Call the callback with the Stock list
                })
                .onFailure(mainDataHandler::handleFailure)
                .fetch();
    }

    public static void getInfoArray(String symbolName, DataCallback callback) {
        Double[] data = new Double[9];

        // Fetch fundamental data
        AlphaVantage.api()
                .fundamentalData()
                .companyOverview()
                .forSymbol(symbolName)
                .onSuccess(e -> {
                    CompanyOverviewResponse companyOverviewResponse = (CompanyOverviewResponse) e;
                    CompanyOverview response = companyOverviewResponse.getOverview();
                    data[4] = response.getPERatio();
                    data[5] = response.getPEGRatio();
                    data[6] = response.getFiftyTwoWeekHigh();
                    data[7] = response.getFiftyTwoWeekLow();
                    data[8] = Double.valueOf(response.getMarketCapitalization());

                })
                .onFailure(mainDataHandler::handleFailure)
                .fetch();

        AlphaVantage.api()
                .timeSeries()
                .quote()
                .forSymbol(symbolName)
                .onSuccess(e -> {
                    QuoteResponse response = (QuoteResponse) e;
                    data[0] = response.getOpen();
                    data[1] = response.getHigh();
                    data[2] = response.getLow();
                    data[3] = response.getVolume();

                    // Call the callback with the fetched data
                    callback.onDataFetched(data);
                })
                .onFailure(mainDataHandler::handleFailure)
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
                            .stream()
                            .map(StockResponse.StockMatch::getSymbol)
                            .toList();
                    callback.onSuccess(allSymbols);
                })
                .onFailure(failure -> {
                    // Handle failure and invoke the failure callback
                    mainDataHandler.handleFailure(failure);
                    callback.onFailure(new RuntimeException("API call failed"));
                })
                .fetch();
    }

    public static void receiveNews(String Symbol, ReceiveNewsCallback callback) {
        AlphaVantage.api()
                .News()
                .setTickers(Symbol)
                .setSort("LATEST")
                .setLimit(12)
                .onSuccess(e -> callback.onNewsReceived(e.getNewsItems()))
                .onFailure(mainDataHandler::handleFailure)
                .fetch();
    }

    public static void startHypeMode(int tradeVolume) {
        logTextArea.append(String.format("Activating hype mode for auto Stock scanning, Settings: %s Volume, %s Stocks to scan\n", tradeVolume, stockSymbols.length));
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
                getAvailableSymbols(tradeVolume, stockSymbols, result -> {
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

    public static void getAvailableSymbols(int tradeVolume, String[] possibleSymbols, SymbolCallback callback) {
        if (possibleSymbols.length == 0) {
            callback.onSymbolsAvailable(Collections.emptyList());
            return;
        }

        List<String> actualSymbols = new CopyOnWriteArrayList<>();
        AtomicInteger remaining = new AtomicInteger(possibleSymbols.length);

        for (String symbol : possibleSymbols) {
            AlphaVantage.api()
                    .fundamentalData()
                    .companyOverview()
                    .forSymbol(symbol)
                    .onSuccess(e -> {
                        CompanyOverviewResponse companyResponse = (CompanyOverviewResponse) e;
                        long marketCapitalization = companyResponse.getOverview().getMarketCapitalization();
                        long sharesOutstanding = companyResponse.getOverview().getSharesOutstanding();

                        AlphaVantage.api()
                                .timeSeries()
                                .daily()
                                .forSymbol(symbol)
                                .outputSize(OutputSize.COMPACT)
                                .onSuccess(tsResponse -> {
                                    TimeSeriesResponse ts = (TimeSeriesResponse) tsResponse;
                                    if (ts.getStockUnits().isEmpty()) {
                                        checkCompletion(remaining, actualSymbols, callback);
                                        return;
                                    }

                                    double close = ts.getStockUnits().get(0).getClose();
                                    double volume = ts.getStockUnits().get(0).getVolume();

                                    if (tradeVolume < marketCapitalization
                                            && ((double) tradeVolume / close) < volume
                                            && ((long) (tradeVolume / close) < sharesOutstanding)) {
                                        actualSymbols.add(symbol);
                                    }

                                    checkCompletion(remaining, actualSymbols, callback);
                                })
                                .onFailure(error -> {
                                    mainDataHandler.handleFailure(error);
                                    checkCompletion(remaining, actualSymbols, callback);
                                })
                                .fetch();
                    })
                    .onFailure(error -> {
                        mainDataHandler.handleFailure(error);
                        checkCompletion(remaining, actualSymbols, callback);
                    })
                    .fetch();
        }
    }

    private static void checkCompletion(AtomicInteger remaining, List<String> actualSymbols, SymbolCallback callback) {
        if (remaining.decrementAndGet() == 0) {
            callback.onSymbolsAvailable(actualSymbols);
        }
    }

    public static void hypeModeFinder(List<String> symbols) {
        logTextArea.append("Started pulling data from server\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        CountDownLatch countDownLatch = new CountDownLatch(symbols.size());

        for (String symbol : symbols) {
            AlphaVantage.api()
                    .timeSeries()
                    .intraday()
                    .forSymbol(symbol)
                    .interval(Interval.ONE_MIN)
                    .outputSize(OutputSize.COMPACT)
                    .onSuccess(e -> {
                        try {
                            TimeSeriesResponse response = (TimeSeriesResponse) e;
                            List<StockUnit> units = response.getStockUnits();

                            units.forEach(stockUnit -> stockUnit.setSymbol(symbol));

                            // Reverse the list to correct chronological order
                            List<StockUnit> reversedUnits = new ArrayList<>(units);
                            Collections.reverse(reversedUnits);

                            // Add reversed units to symbol timeline
                            synchronized (symbolTimelines) {
                                symbolTimelines.computeIfAbsent(symbol, k ->
                                        Collections.synchronizedList(new ArrayList<>())
                                ).addAll(reversedUnits);
                            }

                            if (reversedUnits.isEmpty()) {
                                System.out.println("Empty response for: " + symbol);
                            }

                            countDownLatch.countDown();
                        } catch (Exception ex) {
                            ex.printStackTrace();
                            countDownLatch.countDown();
                        }
                    })
                    .onFailure(error -> {
                        mainDataHandler.handleFailure(error);
                        countDownLatch.countDown();
                    })
                    .fetch();
        }

        try {
            countDownLatch.await(1, TimeUnit.MINUTES);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        synchronized (symbolTimelines) {
            symbolTimelines.forEach((symbol, timeline) -> {
                if (timeline.size() < 2) {
                    logTextArea.append("Not enough data for " + symbol + "\n");
                    return;
                }

                for (int i = 1; i < timeline.size(); i++) {
                    StockUnit current = timeline.get(i);
                    StockUnit previous = timeline.get(i - 1);

                    if (previous.getClose() > 0) {
                        double change = ((current.getClose() - previous.getClose()) / previous.getClose()) * 100;
                        change = Math.abs(change) >= 14 ? previous.getPercentageChange() : change;
                        current.setPercentageChange(change);
                    }
                }
            });
        }

        precomputeIndicatorRanges(true);
        calculateStockPercentageChange(false);

        while (!Thread.currentThread().isInterrupted()) {
            List<RealTimeResponse.RealTimeMatch> matches = new CopyOnWriteArrayList<>();
            try {
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

                // wait 1 Minute
                Thread.sleep(60000);

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
                .onFailure(mainDataHandler::handleFailure)
                .fetch();
    }

    public static void getCompanyOverview(String symbol, OverviewCallback callback) {
        AlphaVantage.api()
                .fundamentalData()
                .companyOverview()
                .forSymbol(symbol)
                .onSuccess(response -> {
                    CompanyOverviewResponse overview = (CompanyOverviewResponse) response;
                    callback.onOverviewReceived(overview);
                })
                .onFailure(mainDataHandler::handleFailure)
                .fetch();
    }

    public static void processStockData(List<RealTimeResponse.RealTimeMatch> matches) {
        Map<String, StockUnit> currentBatch = new ConcurrentHashMap<>();

        for (RealTimeResponse.RealTimeMatch match : matches) {
            String symbol = match.getSymbol().toUpperCase();
            StockUnit unit = new StockUnit.Builder()
                    .symbol(symbol)
                    .close(match.getClose())
                    .time(match.getTimestamp())
                    .volume(match.getVolume())
                    .high(match.getHigh())
                    .open(match.getOpen())
                    .build();

            // Update symbol timeline
            symbolTimelines.computeIfAbsent(symbol, k -> Collections.synchronizedList(new ArrayList<>())).add(unit);
            currentBatch.put(symbol, unit);
        }
        logTextArea.append("Processed " + currentBatch.size() + " valid stock entries\n");
        calculateStockPercentageChange(true);
    }

    public static void calculateStockPercentageChange(boolean realFrame) {
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

        calculateSpikesInRally(frameSize, realFrame);
    }

    public static void calculateSpikesInRally(int minutesPeriod, boolean realFrame) {
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

                LinkedList<StockUnit> timeWindow = new LinkedList<>(getTimeWindow(timeline, startTime, endTime));
                int startIndex = findTimeIndex(timeline, startTime);

                while (timeWindow.size() < frameSize && startIndex > 0) {
                    startIndex--;
                    timeWindow.addFirst(timeline.get(startIndex));
                }

                if (timeWindow.size() >= frameSize) {
                    try {
                        List<Notification> notifications = getNotificationForFrame(timeWindow, symbol);
                        stockNotifications.addAll(notifications);

                        // Add notifications to UI
                        if (!notifications.isEmpty()) {
                            for (Notification notification : notifications) {
                                addNotification(notification.getTitle(), notification.getContent(), notification.getTimeSeries(),
                                        notification.getLocalDateTime(), notification.getSymbol(), notification.getChange(), notification.getConfig());
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        } else {
            // Original notification-based processing
            timeline.forEach(stockUnit -> { //parallel stream is better, but we can't use it since we need to keep the entries in order
                LocalDateTime startTime = stockUnit.getLocalDateTimeDate();
                LocalDateTime endTime = startTime.plusMinutes(minutes);
                int startIndex = findTimeIndex(timeline, startTime);

                List<StockUnit> timeWindow = getTimeWindow(timeline, startTime, endTime);

                // Fallback if not enough data points
                if (timeWindow.size() < frameSize) {
                    int fallbackEnd = Math.min(startIndex + frameSize, timeline.size());
                    timeWindow = timeline.subList(startIndex, fallbackEnd);
                }

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
        notifications.sort(Comparator.comparing(Notification::getLocalDateTime));
    }

    public static void precomputeIndicatorRanges(boolean realData) {
        int maxRequiredPeriod = frameSize;

        List<String> symbolList = new ArrayList<>();
        if (realData) {
            symbolList.addAll(symbolTimelines.keySet());
        } else {
            symbolList.addAll(Arrays.stream(SYMBOLS)
                    .map(s -> s.toUpperCase().replace(".TXT", "")) // map returns transformed elements
                    .toList());
        }

        for (String symbol : symbolList) {
            List<StockUnit> timeline = symbolTimelines.get(symbol);
            if (timeline.size() < maxRequiredPeriod) {
                continue;
            }

            List<String> indicators = new ArrayList<>(INDICATOR_KEYS);
            Map<String, List<Double>> indicatorValues = new HashMap<>();
            indicators.forEach(ind -> indicatorValues.put(ind, new ArrayList<>()));

            // Slide window and collect all feature values
            for (int i = maxRequiredPeriod - 1; i < timeline.size(); i++) {
                List<StockUnit> window = timeline.subList(i - maxRequiredPeriod + 1, i + 1);
                double[] features = computeFeatures(window, symbol);

                for (int j = 0; j < features.length; j++) {
                    String indicator = indicators.get(j);
                    indicatorValues.get(indicator).add(features[j]);
                }
            }

            // Calculate robust min/max using percentiles
            Map<String, Map<String, Double>> symbolRanges = new LinkedHashMap<>();

            for (String indicator : indicators) {
                if (BINARY_INDICATORS.contains(indicator)) {
                    // Hardcoded min/max for binary indicators
                    symbolRanges.put(indicator, Map.of("min", 0.0, "max", 1.0));
                    continue;
                }

                if (indicator.equals("SMA_CROSS")) {
                    // Hardcoded min/max for binary indicators
                    symbolRanges.put(indicator, Map.of("min", -1.0, "max", 1.0));
                    continue;
                }

                List<Double> values = indicatorValues.get(indicator);
                values.sort(Double::compareTo);

                int lowerIndex = (int) (values.size() * 0.01);
                int upperIndex = (int) (values.size() * 0.99);

                double min = values.get(lowerIndex);
                double max = values.get(upperIndex);

                symbolRanges.put(indicator, Map.of("min", min, "max", max));
            }
            SYMBOL_INDICATOR_RANGES.put(symbol, symbolRanges);
        }
    }

    public static double normalizeScore(String indicator, double rawValue, String symbol) {
        Map<String, Map<String, Double>> symbolRanges = SYMBOL_INDICATOR_RANGES.get(symbol);
        Map<String, Double> range = symbolRanges.get(indicator);

        if (range == null) {
            throw new RuntimeException("Empty indicator");
        }

        double min = range.get("min");
        double max = range.get("max");

        if (max == min) return 0.0; // Prevent division by zero

        return switch (indicator) {
            // Binary decision indicators
            case "SMA_CROSS", "KELTNER", "CUMULATIVE_PERCENTAGE" -> rawValue >= 0.5 ? 1.0 : 0.0;
            default -> {
                double normalized = (rawValue - min) / (max - min);
                yield Math.max(0.0, Math.min(1.0, normalized));
            }
        };
    }

    private static double[] computeFeatures(List<StockUnit> stocks, String symbol) {
        double[] features = new double[INDICATOR_KEYS.size()];

        // Trend Following Indicators
        features[0] = isSMACrossover(stocks, 9, 21, symbol); // 0
        features[1] = calculateTRIX(stocks, 5); // 1

        // Momentum Indicators
        features[2] = calculateROC(stocks, 20); // 2

        features[3] = 0.2; // 3

        // Statistical Indicators
        features[4] = isCumulativeSpike(stocks, 10, 0.35); // 4
        features[5] = cumulativePercentageChange(stocks); // 5

        // Advanced Indicators
        features[6] = isKeltnerBreakout(stocks, 12, 10, 0.3, 0.4); // 6
        features[7] = elderRayIndex(stocks, 12); // 7

        return features;
    }

    private static float[] normalizeFeatures(double[] rawFeatures, String symbol) {
        float[] normalizedFeatures = new float[rawFeatures.length];
        List<String> indicatorKeys = new ArrayList<>(INDICATOR_KEYS);
        for (int i = 0; i < rawFeatures.length; i++) {
            normalizedFeatures[i] = (float) normalizeScore(indicatorKeys.get(i), rawFeatures[i], symbol);
        }
        return normalizedFeatures;
    }

    private static double calculateWeightedAggressiveness(float[] features, float baseAggressiveness) {
        Map<String, Double> categoryScores = new HashMap<>();

        // Calculate weighted activation score for each category
        for (int i = 0; i < features.length; i++) {
            String category = FEATURE_CATEGORIES.getOrDefault(i, "NEUTRAL");

            // Dead feature check (very low or 0)
            if (!category.equals("NEUTRAL")) {
                double weight = CATEGORY_WEIGHTS.get(category);
                double activation = features[i] * weight;
                categoryScores.merge(category, activation, Double::sum);
            }
        }

        // Sum all category scores
        double weightedScore = categoryScores.values().stream()
                .mapToDouble(Double::doubleValue)
                .sum();

        // Final aggressiveness scaling
        return baseAggressiveness * (1 + weightedScore);
    }

    public static List<Notification> getNotificationForFrame(List<StockUnit> stocks, String symbol) {
        TimeSeries timeSeries = new TimeSeries(symbol);

        // Prevent notifications if time frame spans over the weekend (Friday to Monday)
        if (isWeekendSpan(stocks)) {
            return new ArrayList<>(); // Return empty list to skip notifications
        }

        //raw features
        double[] features = computeFeatures(stocks, symbol);
        float[] normalizedFeatures = normalizeFeatures(features, symbol);

        // feed normalized features and symbol
        double prediction = predict(normalizedFeatures, symbol);

        synchronized (featureTimeSeriesArray) {
            for (int i = 0; i < features.length; i++) {
                if (i == 3) continue;
                featureTimeSeriesArray[i].addOrUpdate(
                        new Second(stocks.get(stocks.size() - 1).getDateDate()),
                        features[i]
                );
            }
        }

        synchronized (predictionTimeSeries) {
            predictionTimeSeries.addOrUpdate(
                    new Second(stocks.get(stocks.size() - 1).getDateDate()),
                    prediction
            );
        }

        try {
            for (StockUnit stockUnit : stocks) {
                timeSeries.addOrUpdate(new Second(stockUnit.getDateDate()), stockUnit.getClose());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return evaluateResult(timeSeries, prediction, stocks, symbol, features, normalizedFeatures);
    }

    // Method for evaluating results
    private static List<Notification> evaluateResult(TimeSeries timeSeries, double prediction,
                                                     List<StockUnit> stocks, String symbol, double[] features, float[] normalizedFeatures) {
        List<Notification> alertsList = new ArrayList<>();

        double changeUp = stocks.stream().skip(stocks.size() - 4).
                mapToDouble(StockUnit::getPercentageChange).sum();

        double nearRes = isNearResistance(stocks);

        // Dip down
        dipDown(timeSeries, prediction, stocks, symbol, changeUp, alertsList);

        // fill the gap
        fillTheGap(timeSeries, prediction, stocks, symbol, alertsList);

        if (aggressiveness == 0.0) {
            aggressiveness = 1.0F;
        }

        // Spike & R-Line
        spikeUp(timeSeries, prediction, stocks, symbol, features, changeUp, alertsList, nearRes, aggressiveness, normalizedFeatures);

        return alertsList;
    }

    private static void spikeUp(TimeSeries timeSeries, double prediction, List<StockUnit> stocks, String symbol, double[] features, double changeUp,
                                List<Notification> alertsList, double nearRes, float manualAggressiveness, float[] normalizedFeatures) {

        // Calculate dynamic aggressiveness
        double dynamicAggro = calculateWeightedAggressiveness(normalizedFeatures, manualAggressiveness);

        // Adaptive thresholds based on weights
        double cumulativeThreshold = 0.6 * dynamicAggro;

        System.out.println(features[4] +
                ", Mtm: " + String.format("%.3f", features[5]) +
                ", Th: " + String.format("%.3f", cumulativeThreshold) +
                ", Agg: " + String.format("%.3f", dynamicAggro) +
                ", Pred: " + String.format("%.3f", prediction) +
                ", FT6: " + features[6] +
                ", CGU: " + String.format("%.3f", changeUp) +
                " " + stocks.get(stocks.size() - 1).getDateDate());

        if (features[4] == 1 &&
                (features[5] > cumulativeThreshold || dynamicAggro > 2.5) &&
                prediction > 0.9 &&
                features[6] == 1) {

            if (nearRes == 0) {
                createNotification(symbol, changeUp, alertsList, timeSeries,
                        stocks.get(stocks.size() - 1).getLocalDateTimeDate(),
                        prediction, 3);
            } else {
                createNotification(symbol, changeUp, alertsList, timeSeries,
                        stocks.get(stocks.size() - 1).getLocalDateTimeDate(),
                        prediction, 2);
            }
        }
    }

    private static void fillTheGap(TimeSeries timeSeries, double prediction, List<StockUnit> stocks,
                                   String symbol, List<Notification> alertsList) {
        // Adaptive parameters
        int smaPeriod = 20;
        int atrPeriod = 14;
        int rsiPeriod = 10;
        int stochasticPeriod = 14;
        int dropLookBack = 10;
        int stochasticLimit = 5;
        int rsiLimit = 32;
        double sharpDropThreshold = 2.0;
        double dipThreshold = -2.5;

        if (stocks.size() >= smaPeriod) {
            int endIndex = stocks.size();
            int startIndex = endIndex - smaPeriod;

            // 1. Enhanced Trend Detection
            double sma = calculateSMA(stocks.subList(startIndex, endIndex));
            double currentClose = stocks.get(endIndex - 1).getClose();
            double deviation = currentClose - sma;

            // 2. Dynamic Gap Threshold
            double atr = calculateATR(stocks.subList(endIndex - atrPeriod, endIndex), atrPeriod);
            double historicalATR = getHistoricalATR(symbol);
            double volatilityRatio = atr / historicalATR;

            // Adaptive multiplier (more sensitive in high volatility)
            double multiplier = Math.max(1.5, 3.0 - (volatilityRatio * 0.8));
            double gapThreshold = -multiplier * atr;

            // 4. Momentum Confirmation
            double rsi = calculateRSI(stocks, rsiPeriod);
            boolean oversold = rsi < rsiLimit;
            double stochastic = calculateStochastic(stocks, stochasticPeriod);
            boolean momentumDivergence = stochastic < stochasticLimit;

            // 5. Enhanced Drop Detection
            boolean sharpDrop = checkSharpDrop(stocks, dropLookBack, sharpDropThreshold);

            // 6. Wide Gap Specific Checks
            boolean isWideGap = deviation < (gapThreshold * 1.3);
            boolean sustainedMove = checkSustainedMovement(stocks, dropLookBack, dipThreshold);

            // Signal Generation Logic
            boolean baseCondition =
                    (deviation < gapThreshold || isWideGap) &&
                            (sharpDrop || sustainedMove) &&
                            (oversold || momentumDivergence);

            System.out.println(
                    " | Deviation: " + String.format("%.3f", deviation) +
                            " | ATR: " + String.format("%.3f", atr) +
                            " | VolatilityRatio: " + String.format("%.3f", volatilityRatio) +
                            " | Multiplier: " + String.format("%.3f", multiplier) +
                            " | GapThreshold: " + String.format("%.3f", gapThreshold) +
                            " | RSI: " + String.format("%.3f", rsi) +
                            " | Oversold: " + oversold +
                            " | Stochastic: " + String.format("%.3f", stochastic) +
                            " | MomentumDivergence: " + momentumDivergence +
                            " | SharpDrop: " + sharpDrop +
                            " | IsWideGap: " + isWideGap +
                            " | SustainedMove: " + sustainedMove +
                            " | base: " + baseCondition +
                            " | Date: " + stocks.get(stocks.size() - 1).getDateDate());

            if (baseCondition) {
                createNotification(symbol, deviation, alertsList, timeSeries,
                        stocks.get(endIndex - 1).getLocalDateTimeDate(),
                        prediction, 1);
            }
        }
    }

    //Indicators
    private static double isNearResistance(List<StockUnit> stocks) {
        if (stocks.size() < 2) {
            return 0.0; // Not enough data points
        }

        // Exclude last candle to determine resistance level
        List<StockUnit> previousStocks = stocks.subList(0, stocks.size() - 1);

        if (previousStocks.isEmpty()) {
            return 0.0;
        }

        double resistanceLevel = previousStocks.stream()
                .mapToDouble(StockUnit::getHigh)
                .max()
                .orElse(0.0);

        StockUnit currentStock = stocks.get(stocks.size() - 1);
        double currentClose = currentStock.getClose();
        double threshold = resistanceLevel * 0.995; // Within 0.5% below resistance

        return (currentClose >= threshold && currentClose <= resistanceLevel) ? 1.0 : 0.0;
    }

    private static void dipDown(TimeSeries timeSeries, double prediction, List<StockUnit> stocks,
                                String symbol, double changeUp, List<Notification> alertsList) {
        // Sensitivity Configuration
        int windowSize = 8;      // Look at 5-day window for patterns
        double atrMultiplier = 1.2;  // More sensitive threshold
        double minDropPct = 4.0;    // Minimum total percentage drop
        int minDownDays = 3;      // At least 3 down days in window

        if (stocks.size() < windowSize + 1) return;

        List<StockUnit> window = stocks.subList(stocks.size() - windowSize - 1, stocks.size());
        double atr = calculateATR(window, 14);

        // Track pattern characteristics
        int downDays = 0;
        double cumulativeDrop = 0;
        double maxClose = Double.MIN_VALUE;
        double minClose = Double.MAX_VALUE;

        for (int i = 1; i < window.size(); i++) {
            StockUnit current = window.get(i);
            StockUnit previous = window.get(i - 1);

            // Track price extremes
            maxClose = Math.max(maxClose, previous.getClose());
            minClose = Math.min(minClose, current.getClose());

            if (current.getClose() < previous.getClose()) {
                downDays++;
                cumulativeDrop += (previous.getClose() - current.getClose());
            }
        }

        // Calculate intensity metrics
        double totalDropPct = ((maxClose - minClose) / maxClose) * 100;

        // Recent bounce check (last 2 periods)
        boolean hasBounce = window.get(window.size() - 1).getClose() >
                window.get(window.size() - 2).getLow();

        // Activation Conditions (all must be true)
        boolean isSensitiveDip =
                downDays >= minDownDays &&
                        cumulativeDrop >= (atr * atrMultiplier) &&
                        totalDropPct >= minDropPct &&
                        hasBounce;

        if (isSensitiveDip) {
            createNotification(symbol, changeUp, alertsList, timeSeries,
                    stocks.get(stocks.size() - 1).getLocalDateTimeDate(),
                    prediction, 0);
        }
    }

    // 1. Simple Moving Average (SMA)
    public static int isSMACrossover(List<StockUnit> window, int shortPeriod, int longPeriod, String symbol) {
        if (window == null || window.size() < longPeriod + 1 || shortPeriod >= longPeriod) {
            return smaStateMap.getOrDefault(symbol, 0); // Return current state if invalid
        }

        // Extract closing prices once
        double[] closes = window.stream()
                .mapToDouble(StockUnit::getClose)
                .toArray();

        // Calculate current SMAs
        double shortSMA = calculateSMA(closes, closes.length - shortPeriod, shortPeriod);
        double longSMA = calculateSMA(closes, closes.length - longPeriod, longPeriod);

        // Calculate previous SMAs (one period back)
        double prevShortSMA = calculateSMA(closes, closes.length - shortPeriod - 1, shortPeriod);
        double prevLongSMA = calculateSMA(closes, closes.length - longPeriod - 1, longPeriod);

        boolean bullishCrossover = (prevShortSMA <= prevLongSMA) && (shortSMA > longSMA);
        boolean bearishCrossover = (prevShortSMA >= prevLongSMA) && (shortSMA < longSMA);

        int currentState = smaStateMap.getOrDefault(symbol, 0);

        if (bullishCrossover) {
            smaStateMap.put(symbol, 1);
            currentState = 1;
        } else if (bearishCrossover) {
            smaStateMap.put(symbol, -1);
            currentState = -1;
        }

        return currentState;
    }

    // 2. TRIX Indicator
    public static double calculateTRIX(List<StockUnit> prices, int period) {
        final int minDataPoints = 3 * period + 1;
        if (prices.size() < minDataPoints || period < 2) {
            return 0; // Not enough data for valid calculation
        }

        // 1. Extract closing prices in chronological order (oldest first)
        List<Double> closes = prices.stream()
                .map(StockUnit::getClose)
                .collect(Collectors.toList());

        // 2. Calculate triple-smoothed EMA
        List<Double> singleEMA = calculateEMASeries(closes, period);
        List<Double> doubleEMA = calculateEMASeries(singleEMA, period);
        List<Double> tripleEMA = calculateEMASeries(doubleEMA, period);

        // 3. Calculate rate of change for TRIX
        if (tripleEMA.size() < 2) return 0;

        double current = tripleEMA.get(tripleEMA.size() - 1);
        double previous = tripleEMA.get(tripleEMA.size() - 2);

        return ((current - previous) / previous) * 100;
    }

    // 3. Rate of Change (ROC) with SIMD optimization
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

    // 4. Cumulative Percentage Change with threshold check
    public static int isCumulativeSpike(List<StockUnit> window, int period, double threshold) {
        if (window.size() < period) return 0;

        double sum = window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getPercentageChange)
                .sum();

        // Return 1 if true, 0 if false
        return sum >= threshold ? 1 : 0;
    }

    // 5. Cumulative Percentage Change
    private static double cumulativePercentageChange(List<StockUnit> stocks) {
        int startIndex = 0;
        try {
            startIndex = stocks.size() - 8;
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Using Stream to process the list
        return stocks.subList(startIndex, stocks.size())
                .stream()
                .mapToDouble(StockUnit::getPercentageChange)  // Convert to double
                .sum();  // Sum all the results
    }

    // 6. Keltner Channels Breakout
    public static int isKeltnerBreakout(List<StockUnit> window, int emaPeriod, int atrPeriod, double multiplier,
                                        double cumulativeLimit) {
        // Check if we have enough data for calculations
        if (window.size() < Math.max(emaPeriod, 4) + 1) {
            return 0; // Not enough data points
        }

        // Original Keltner Channel calculation
        double ema = calculateEMA(window, emaPeriod);
        double atr = calculateATR(window, atrPeriod);
        double upperBand = ema + (multiplier * atr);

        // Cumulative percentage change check
        int currentIndex = window.size() - 1;
        int referenceIndex = currentIndex - 8;

        double currentClose = window.get(currentIndex).getClose();
        double referenceClose = window.get(referenceIndex).getClose();

        double cumulativeChange = ((currentClose - referenceClose) / referenceClose) * 100;

        // Combined condition check
        boolean isBreakout = currentClose > upperBand;
        boolean hasSignificantMove = Math.abs(cumulativeChange) >= cumulativeLimit;

        return (isBreakout && hasSignificantMove) ? 1 : 0;
    }

    // 7. Elder-Ray Index Approximation
    public static double elderRayIndex(List<StockUnit> window, int emaPeriod) {
        double ema = calculateEMA(window, emaPeriod);
        return window.get(window.size() - 1).getClose() - ema;
    }

    private static double calculateSMA(double[] closes, int startIndex, int period) {
        if (startIndex < 0 || startIndex + period > closes.length) return 0;

        double sum = 0;
        for (int i = startIndex; i < startIndex + period; i++) {
            sum += closes[i];
        }
        return sum / period;
    }

    private static double calculateEMA(List<StockUnit> prices, int period) {
        if (prices.size() < period || period <= 0) {
            return 0; // Handle invalid input
        }

        // 1. Calculate SMA for first period values
        double sma = prices.stream()
                .limit(period)
                .mapToDouble(StockUnit::getClose)
                .average()
                .orElse(0);

        // 2. Calculate smoothing factor
        final double smoothing = 2.0 / (period + 1);
        double ema = sma;

        // 3. Apply EMA formula to subsequent values
        for (int i = period; i < prices.size(); i++) {
            double currentPrice = prices.get(i).getClose();
            ema = (currentPrice - ema) * smoothing + ema;
        }

        return ema;
    }

    private static List<Double> calculateEMASeries(List<Double> data, int period) {
        if (data.size() < period) return Collections.emptyList();

        List<Double> emaSeries = new ArrayList<>();
        double smoothing = 2.0 / (period + 1);

        // Initial SMA calculation
        double sma = data.subList(0, period)
                .stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0);
        emaSeries.add(sma);

        // Subsequent EMA calculations
        for (int i = period; i < data.size(); i++) {
            double currentValue = data.get(i);
            double prevEMA = emaSeries.get(emaSeries.size() - 1);
            double newEMA = (currentValue - prevEMA) * smoothing + prevEMA;
            emaSeries.add(newEMA);
        }

        return emaSeries;
    }

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

    private static double calculateRSI(List<StockUnit> stocks, int period) {
        if (stocks.size() <= period || period < 1) return 50; // Neutral default

        List<Double> changes = new ArrayList<>();
        for (int i = 1; i <= period; i++) {
            double change = stocks.get(i).getClose() - stocks.get(i - 1).getClose();
            changes.add(change);
        }

        double avgGain = changes.stream().filter(c -> c > 0).mapToDouble(Double::doubleValue).average().orElse(0);
        double avgLoss = Math.abs(changes.stream().filter(c -> c < 0).mapToDouble(Double::doubleValue).average().orElse(0));

        if (avgLoss == 0) return 100; // Prevent division by zero
        double rs = avgGain / avgLoss;
        return 100 - (100 / (1 + rs));
    }

    private static double calculateSMA(List<StockUnit> periodStocks) {
        if (periodStocks.isEmpty()) return 0;
        return periodStocks.stream()
                .mapToDouble(StockUnit::getClose)
                .average()
                .orElse(0);
    }

    private static double getHistoricalATR(String symbol) {
        // Return cached value if available
        if (historicalATRCache.containsKey(symbol)) {
            return historicalATRCache.get(symbol);
        }

        List<StockUnit> allStocks = symbolTimelines.get(symbol);
        if (allStocks == null || allStocks.isEmpty()) {
            return 1.0; // Fallback default
        }

        int availableDays = allStocks.size();
        int calculatedLookback = Math.min(availableDays, HISTORICAL_LOOK_BACK);

        if (calculatedLookback < 14) {
            return 1.0; // Insufficient data for reliable calculation
        }

        // Extract relevant historical data
        int startIndex = Math.max(0, availableDays - calculatedLookback);
        List<StockUnit> historicalData = allStocks.subList(startIndex, availableDays);

        // Calculate ATR using full lookback period
        double atr = calculateATR(historicalData, calculatedLookback);

        // Cache and return
        historicalATRCache.put(symbol, atr);
        return atr;
    }

    private static boolean checkSharpDrop(List<StockUnit> stocks, int lookBack, double threshold) {
        if (stocks.size() < lookBack + 1) return false;

        double currentClose = stocks.get(stocks.size() - 1).getClose();

        for (int i = 1; i <= lookBack; i++) {
            double previousClose = stocks.get(stocks.size() - 1 - i).getClose();
            if (((previousClose - currentClose) / previousClose) * 100 > threshold) {
                return true;
            }
        }
        return false;
    }

    private static boolean checkSustainedMovement(List<StockUnit> stocks, int lookback, double threshold) {
        if (stocks.size() < lookback) return false;

        double totalMove = 0;
        for (int i = 1; i <= lookback; i++) {
            StockUnit current = stocks.get(stocks.size() - i);
            StockUnit previous = stocks.get(stocks.size() - i - 1);
            totalMove += (current.getClose() - previous.getClose()) / previous.getClose();
        }
        return totalMove * 100 < threshold;
    }

    private static double calculateStochastic(List<StockUnit> stocks, int period) {
        // Implement stochastic oscillator
        List<StockUnit> lookback = stocks.subList(stocks.size() - period, stocks.size());
        double highestHigh = lookback.stream().mapToDouble(StockUnit::getHigh).max().orElse(0);
        double lowestLow = lookback.stream().mapToDouble(StockUnit::getLow).min().orElse(0);
        double lastClose = lookback.get(lookback.size() - 1).getClose();

        if (highestHigh == lowestLow) return 50;
        return 100 * (lastClose - lowestLow) / (highestHigh - lowestLow);
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
     * config 0 dip
     * config 1 gap filler
     * config 2 R-line spike
     * config 3 spike
     *
     * @param symbol      The name of the stock.
     * @param totalChange The total percentage change triggering the notification.
     * @param alertsList  The list to store the notification.
     * @param timeSeries  The time series for graphical representation.
     * @param date        The date of the event.
     */
    private static void createNotification(String symbol, double totalChange, List<
            Notification> alertsList, TimeSeries timeSeries, LocalDateTime date, double prediction, int config) {
        if (config == 0) {
            alertsList.add(new Notification(String.format("%.3f%% %s ↓ %.3f, %s", totalChange, symbol, prediction, date.format(DateTimeFormatter.ofPattern("HH:mm:ss"))),
                    String.format("Decreased by %.3f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))),
                    timeSeries, date, symbol, totalChange, 0));
        } else if (config == 1) {
            alertsList.add(new Notification(String.format("Gap fill %s ↓↑ %.3f, %s", symbol, prediction, date.format(DateTimeFormatter.ofPattern("HH:mm:ss"))),
                    String.format("Will fill the gap at the %s", date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))),
                    timeSeries, date, symbol, totalChange, 1));
        } else if (config == 2) {
            alertsList.add(new Notification(String.format("%.3f%% %s R-Line %.3f, %s", totalChange, symbol, prediction, date.format(DateTimeFormatter.ofPattern("HH:mm:ss"))),
                    String.format("R-Line Spike Proceed with caution by %.3f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))),
                    timeSeries, date, symbol, totalChange, 2));
        } else if (config == 3) {
            alertsList.add(new Notification(String.format("%.3f%% %s ↑ %.3f, %s", totalChange, symbol, prediction, date.format(DateTimeFormatter.ofPattern("HH:mm:ss"))),
                    String.format("Increased by %.3f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))),
                    timeSeries, date, symbol, totalChange, 3));
        }
    }

    public static List<StockUnit> getSymbolTimeline(String symbol) {
        return Collections.unmodifiableList(
                symbolTimelines.getOrDefault(symbol.toUpperCase(), new ArrayList<>())
        );
    }

    public static void realTimeDataCollector(String symbol) {
        InitAPi("0988PSIKXZ50IP2T");
        executorService.scheduleAtFixedRate(() -> getRealTimeUpdate(symbol, response -> {
            try {
                File data = new File("realtime.txt");
                if (!data.exists()) {
                    data.createNewFile();
                }

                // Use try-with-resources for automatic resource management
                try (BufferedWriter writer = new BufferedWriter(new FileWriter(data, true))) { // true for append mode

                    String formatted = String.format(
                            "StockUnit{open=%.4f, high=%.4f, low=%.4f, close=%.4f, " +
                                    "adjustedClose=%.1f, volume=%.1f, dividendAmount=%.1f, " +
                                    "splitCoefficient=%.1f, date=%s, symbol=%s, " +
                                    "percentageChange=%.1f, target=%d},",
                            response.getOpen(),
                            response.getHigh(),
                            response.getLow(),
                            response.getClose(),
                            0.0,
                            response.getVolume(),
                            0.0,
                            0.0,
                            response.getTimestamp(),
                            response.getSymbol(),
                            0.0,
                            0
                    );

                    writer.write(formatted);
                    writer.newLine();

                }  // BufferedWriter auto-closes here

            } catch (IOException e) {
                System.err.println("Error writing to file: " + e.getMessage());
            }
        }), 0, 1, TimeUnit.SECONDS);
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

    public interface OverviewCallback {
        void onOverviewReceived(CompanyOverviewResponse value);
    }
}