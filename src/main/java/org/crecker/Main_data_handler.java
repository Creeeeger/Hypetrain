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

import java.io.*;
import java.time.DayOfWeek;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.crecker.Main_UI.addNotification;
import static org.crecker.Main_UI.logTextArea;
import static org.crecker.RallyPredictor.predict;

public class Main_data_handler {
    public static final Map<String, Map<String, Double>> INDICATOR_RANGE_MAP = new LinkedHashMap<>() {{
        // Trend Following Indicators
        put("SMA_CROSS", Map.of("min", -1.0, "max", 1.0));
        put("TRIX", Map.of("min", -0.5, "max", 0.5));

        // Momentum Indicators
        put("ROC", Map.of("min", -5.0, "max", 5.0));

        // Volatility & Breakouts Indicators
        put("BOLLINGER", Map.of("min", 0.0, "max", 0.1));

        // Statistical Indicators
        put("CUMULATIVE_PERCENTAGE", Map.of("min", 0.0, "max", 1.0));
        put("CUMULATIVE_THRESHOLD", Map.of("min", -7.0, "max", 7.0));

        // Advanced Indicators
        put("KELTNER", Map.of("min", 0.0, "max", 1.0));
        put("ELDER_RAY", Map.of("min", -3.0, "max", 3.0));
    }};
    static final Map<String, List<StockUnit>> symbolTimelines = new HashMap<>();
    static final List<Notification> notificationsForPLAnalysis = new ArrayList<>();
    static final TimeSeries indicatorTimeSeries = new TimeSeries("Indicator levels");
    static final TimeSeries predictionTimeSeries = new TimeSeries("Predictions");
    private static final ConcurrentHashMap<String, Integer> smaStateMap = new ConcurrentHashMap<>();
    static int frameSize = 30; // Frame size for analysis

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

    public static void start_Hype_Mode(int tradeVolume) {
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

        calculateSpikesInRally(frameSize, true);
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

                List<StockUnit> timeWindow = getTimeWindow(timeline, startTime, endTime);

                if (timeWindow.size() >= frameSize) {
                    try {
                        List<Notification> notifications = getNotificationForFrame(timeWindow, symbol);
                        stockNotifications.addAll(notifications);

                        // Add notifications to UI
                        if (!notifications.isEmpty()) {
                            for (Notification notification : notifications) {
                                addNotification(notification.getTitle(), notification.getContent(), notification.getTimeSeries(),
                                        notification.getLocalDateTime(), notification.getSymbol(), notification.getChange());
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
        if (range == null) return rawValue;

        double min = range.get("min");
        double max = range.get("max");

        if (max == min) return 0.0; // Prevent division by zero

        return switch (indicator) {
            // Binary decision indicators
            case "SMA_CROSS", "KELTNER", "CUMULATIVE_PERCENTAGE" -> rawValue >= 0.5 ? 1.0 : 0.0;

            // Standard continuous indicators with linear normalization
            case "TRIX", "ROC", "BOLLINGER", "ELDER_RAY", "CUMULATIVE_THRESHOLD" -> {
                double normalized = (rawValue - min) / (max - min);
                yield Math.max(0.0, Math.min(1.0, normalized));
            }
            default -> throw new IllegalStateException("Unexpected value: " + indicator);
        };
    }

    private static double[] computeFeatures(List<StockUnit> stocks, String symbol) {
        double[] features = new double[INDICATOR_RANGE_MAP.size()];
        int featureIndex = 0;

        // Trend Following Indicators
        features[featureIndex++] = isSMACrossover(stocks, 9, 21, symbol); // 0
        features[featureIndex++] = calculateTRIX(stocks, 5); // 1

        // Momentum Indicators
        features[featureIndex++] = calculateROC(stocks, 20); // 2

        // Volatility & Breakouts Indicators
        features[featureIndex++] = calculateBollingerBands(stocks, 20); // 3

        // Statistical Indicators
        features[featureIndex++] = isCumulativeSpike(stocks, 10, 0.35); // 4
        features[featureIndex++] = cumulativePercentageChange(stocks); // 5

        // Advanced Indicators
        features[featureIndex++] = isKeltnerBreakout(stocks, 12, 10, 0.3, 0.4); // 6
        features[featureIndex++] = elderRayIndex(stocks, 12); // 7

        return features;
    }

    private static float[] normalizeFeatures(double[] rawFeatures) {
        float[] normalizedFeatures = new float[rawFeatures.length];
        List<String> indicatorKeys = new ArrayList<>(INDICATOR_RANGE_MAP.keySet());
        for (int i = 0; i < rawFeatures.length; i++) {
            normalizedFeatures[i] = (float) normalizeScore(indicatorKeys.get(i), rawFeatures[i]);
        }
        return normalizedFeatures;
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

        //raw features
        double[] features = computeFeatures(stocks, symbol);

        // feed normalized features
        double prediction = predict(normalizeFeatures(features));

        synchronized (indicatorTimeSeries) {
            indicatorTimeSeries.addOrUpdate(
                    new Minute(stocks.get(stocks.size() - 1).getDateDate()),
                    features[2]
            );

            predictionTimeSeries.addOrUpdate(
                    new Minute(stocks.get(stocks.size() - 1).getDateDate()),
                    prediction
            );
        }

        for (StockUnit stockUnit : stocks) {
            timeSeries.add(new Minute(stockUnit.getDateDate()), stockUnit.getClose());
        }

        return evaluateResult(timeSeries, prediction, stocks, symbol, features);
    }

    // Method for evaluating results
    private static List<Notification> evaluateResult(TimeSeries timeSeries, double prediction,
                                                     List<StockUnit> stocks, String symbol,
                                                     double[] features) {
        List<Notification> alertsList = new ArrayList<>();

        // 0. isSMACrossover
        // 1. calculateTRIX
        // 2. calculateROC              GD
        // 3. calculateBollingerBands //BS remove //represents other indicators
        // 4. isCumulativeSpike
        // 5. cumulativePercentageChange
        // 6. isKeltnerBreakout
        // 7. elderRayIndex

        if (features[0] == 1) {
            if (features[1] > 0.12) { //maybe >0
                if (features[2] > 0.2) { // > 0.2 +-
                    if (features[6] == 1) {
                        if (features[7] > 0.18) {
                                createNotification(symbol, stocks.stream()
                                        .skip(stocks.size() - 4)
                                        .mapToDouble(StockUnit::getPercentageChange)
                                        .sum(), alertsList, timeSeries, stocks.get(stocks.size() - 1).getLocalDateTimeDate(), prediction);
                        }
                    }
                }
            }
        }

        return alertsList;
    }

    //Indicators
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

    // 4. Bollinger Bands with Bandwidth Expansion
    public static Double calculateBollingerBands(List<StockUnit> window, int period) {
        if (window.size() < period) return 0.0;

        DoubleSummaryStatistics stats = window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getClose)
                .summaryStatistics();

        double sma = stats.getAverage();
        double stdDev = Math.sqrt(window.stream()
                .skip(window.size() - period)
                .mapToDouble(su -> Math.pow(su.getClose() - sma, 2))
                .sum() / period);

        return ((sma + 2 * stdDev) - (sma - 2 * stdDev)) / sma;
    }

    // 5. Cumulative Percentage Change with threshold check
    public static int isCumulativeSpike(List<StockUnit> window, int period, double threshold) {
        if (window.size() < period) return 0;

        double sum = window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getPercentageChange)
                .sum();

        // Return 1 if true, 0 if false
        return sum >= threshold ? 1 : 0;
    }

    // 6. Cumulative Percentage Change
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

    // 7. Keltner Channels Breakout
    public static int isKeltnerBreakout(List<StockUnit> window, int emaPeriod, int atrPeriod, double multiplier, double cumulative_Limit) {
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
        boolean hasSignificantMove = Math.abs(cumulativeChange) >= cumulative_Limit;

        return (isBreakout && hasSignificantMove) ? 1 : 0;
    }

    // 8. Elder-Ray Index Approximation
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
     */
    private static void createNotification(String symbol, double totalChange, List<Notification> alertsList, TimeSeries timeSeries, LocalDateTime date, double prediction) {
        alertsList.add(new Notification(String.format("%.3f%% %s ↑ %s", totalChange, symbol, prediction),
                String.format("Increased by %.3f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))),
                timeSeries, date, symbol, totalChange));
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
}