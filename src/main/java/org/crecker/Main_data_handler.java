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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.crecker.Main_UI.logTextArea;
import static org.crecker.RallyPredictor.predict;
import static org.crecker.csvDataGen.saveFeaturesToCSV;
import static org.crecker.data_tester.plotData;
import static org.crecker.weightRangeMap.*;

public class Main_data_handler {
    static final Map<String, List<StockUnit>> symbolTimelines = new HashMap<>();
    static final List<Notification> notificationsForPLAnalysis = new ArrayList<>();
    static final TimeSeries indicatorTimeSeries = new TimeSeries("Indicator levels");
    public static boolean test = true; // If True use demo url for real Time Updates
    static int feature = 0;
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

        plotData(indicatorTimeSeries, " Indicator", "Time", "Value");
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
        if (range == null) return rawValue;

        double min = range.get("min");
        double max = range.get("max");

        if (max == min) return 0.0; // Prevent division by zero

        return switch (indicator) {
            // Binary decision indicators (threshold at 0.5)
            case "SMA_CROSS", "HIGHER_HIGHS", "TRENDLINE", "PARABOLIC", "KELTNER" -> rawValue >= 0.5 ? 1.0 : 0.0;

            // Continuous percentage indicators (already in 0-1 range)
            case "PRICE_SMA_DISTANCE", "CUMULATIVE_PERCENTAGE" -> Math.max(0.0, Math.min(1.0, rawValue));

            // Standard continuous indicators with linear normalization
            case "MACD", "TRIX", "RSI", "ROC", "MOMENTUM", "CMO",
                 "BOLLINGER", "CONSECUTIVE_POSITIVE_CLOSES", "ELDER_RAY", "ATR", "CUMULATIVE_THRESHOLD" -> {
                double normalized = (rawValue - min) / (max - min);
                yield Math.max(0.0, Math.min(1.0, normalized));
            }
            default -> throw new IllegalStateException("Unexpected value: " + indicator);
        };
    }

    private static double[] computeFeatures(List<StockUnit> stocks) {
        double[] features = new double[INDICATOR_RANGE_MAP.size()];
        int featureIndex = 0;

        // Trend Following Indicators
        features[featureIndex++] = isSMACrossover(stocks, 9, 21); // 1
        features[featureIndex++] = isPriceAboveSMA(stocks, 20); // 2
        features[featureIndex++] = calculateMACD(stocks, 6, 13, 5); // 3
        features[featureIndex++] = calculateTRIX(stocks, 5); // 4

        // Momentum Indicators
        features[featureIndex++] = calculateRSI(stocks, 15); // 5
        features[featureIndex++] = calculateROC(stocks, 20); // 6
        features[featureIndex++] = calculateMomentum(stocks, 10); // 7
        features[featureIndex++] = calculateCMO(stocks, 20); // 8

        // Volatility & Breakouts Indicators
        features[featureIndex++] = calculateBollingerBands(stocks, 20); // 9

        // Patterns Indicators
        features[featureIndex++] = consecutivePositiveCloses(stocks, 0.2); // 10
        features[featureIndex++] = isHigherHighs(stocks, 3); // 11
        features[featureIndex++] = isTrendLineBreakout(stocks, 20); // 12

        // Statistical Indicators
        features[featureIndex++] = isCumulativeSpike(stocks, 10, 0.55); // 13
        features[featureIndex++] = cumulativePercentageChange(stocks); // 14

        // Advanced Indicators
        features[featureIndex++] = isParabolicSARBullish(stocks, 20, 0.01); // 15
        features[featureIndex++] = isKeltnerBreakout(stocks, 20, 20, 0.2); // 16
        features[featureIndex++] = elderRayIndex(stocks, 12); // 17
        features[featureIndex++] = calculateATR(stocks, 20); // 18

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

        Map<String, Double> aggregatedWeights = new HashMap<>();

        INDICATOR_RANGE_FULL.forEach((category, indicators) -> {
            double categoryWeight = INDICATOR_WEIGHTS_FULL.getOrDefault(category, 0.0);
            indicators.forEach((indicator, weight) -> {
                double finalWeight = weight * categoryWeight;
                aggregatedWeights.merge(indicator, finalWeight, Double::sum);
            });
        });

        //raw features
        double[] features = computeFeatures(stocks);

        //normalized features
        float[] normalized = normalizeFeatures(features);

        //weighted list
        double[] weightedFeatures = new double[normalized.length];

        // Map indicators to feature index
        Map<String, Integer> indicatorToIndex = new HashMap<>();
        int idx = 0;
        for (String indicator : INDICATOR_RANGE_MAP.keySet()) {
            indicatorToIndex.put(indicator, idx++);
        }

        // Apply weights
        aggregatedWeights.forEach((indicator, weight) -> {
            Integer index = indicatorToIndex.get(indicator);
            if (index != null && index < normalized.length) {
                weightedFeatures[index] = normalized[index] * weight;
            }
        });

        //feed normalized unweighted features
        double prediction = predict(normalized);

        synchronized (indicatorTimeSeries) {
            indicatorTimeSeries.addOrUpdate(
                    new Minute(stocks.get(stocks.size() - 1).getDateDate()),
                    features[feature]
            );
        }

        for (StockUnit stockUnit : stocks) {
            timeSeries.add(new Minute(stockUnit.getDateDate()), stockUnit.getClose());
        }

        if (pLTester.debug) {
            saveFeaturesToCSV(normalized, INDICATOR_RANGE_MAP, stocks.get(stocks.size() - 1).getDateDate(), prediction);
        }

        return evaluateResult(timeSeries, weightedFeatures, prediction, aggregatedWeights, stocks, symbol, features);
    }

    // Method for evaluating results
    private static List<Notification> evaluateResult(TimeSeries timeSeries, double[] weightedFeatures, double prediction, Map<String, Double> aggregatedWeights, List<StockUnit> stocks, String symbol, double[] features) {
        List<Notification> alertsList = new ArrayList<>();

        if (pLTester.debug) {
            int i = 0;
            for (Map.Entry<String, Double> entry : aggregatedWeights.entrySet()) {
                if (i < weightedFeatures.length) {
                    System.out.println("Key: " + entry.getKey() + ", Weighted Feature: " + weightedFeatures[i] + " " + features[i]);
                }

                i++;
            }
        }

        createNotification(symbol, stocks.stream()
                .skip(stocks.size() - 4)
                .mapToDouble(StockUnit::getPercentageChange)
                .sum(), alertsList, timeSeries, stocks.get(stocks.size() - 1).getLocalDateTimeDate(), prediction);

        return alertsList;
    }

    //Indicators
    // 1. Simple Moving Average (SMA)
    public static int isSMACrossover(List<StockUnit> window, int shortPeriod, int longPeriod) {
        // Validate inputs
        if (window == null || window.size() < longPeriod + 1 || shortPeriod >= longPeriod) {
            return 0;
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

        // Detect crossover with direction check
        return (prevShortSMA <= prevLongSMA && shortSMA > longSMA) ? 1 : 0;
    }

    private static double calculateSMA(double[] closes, int startIndex, int period) {
        if (startIndex < 0 || startIndex + period > closes.length) return 0;

        double sum = 0;
        for (int i = startIndex; i < startIndex + period; i++) {
            sum += closes[i];
        }
        return sum / period;
    }

    // 2. Price Crossing Key Moving Average
    public static int isPriceAboveSMA(List<StockUnit> window, int period) {
        // Validate inputs
        if (window == null || period <= 0 || window.size() < period + 1) {
            return 0;
        }

        // Calculate SMA of previous 'period' closes (excluding current price)
        double sum = 0.0;
        int startIdx = window.size() - period - 1;
        int endIdx = window.size() - 2; // Exclude last element

        for (int i = startIdx; i <= endIdx; i++) {
            sum += window.get(i).getClose();
        }
        double sma = sum / period;

        // Get current and previous prices
        double currentClose = window.get(window.size() - 1).getClose();

        // Detect valid crossover (strict boundary checks)
        return (currentClose > sma) ? 1 : 0;
    }

    // 3. MACD with Histogram
    public static double calculateMACD(List<StockUnit> window, int SHORT_EMA, int LONG_EMA, int SIGNAL_EMA) {
        // Validate inputs
        if (window == null || window.size() < LONG_EMA + SIGNAL_EMA) {
            return 0.0;
        }

        // Extract closing prices once
        List<Double> closes = window.stream()
                .map(StockUnit::getClose)
                .collect(Collectors.toList());

        // 1. Calculate EMA series
        List<Double> shortEMAs = computeEMASeries(closes, SHORT_EMA);
        List<Double> longEMAs = computeEMASeries(closes, LONG_EMA);

        // 2. Calculate MACD line (ensure aligned lengths)
        int minLength = Math.min(shortEMAs.size(), longEMAs.size());
        List<Double> macdLine = new ArrayList<>(minLength);
        for (int i = 0; i < minLength; i++) {
            macdLine.add(shortEMAs.get(i) - longEMAs.get(i));
        }

        // 3. Calculate Signal line (EMA of MACD line)
        List<Double> signalLine = computeEMASeries(macdLine, SIGNAL_EMA);

        // 4. Get latest values with boundary checks
        double currentMACD = macdLine.isEmpty() ? 0 : macdLine.get(macdLine.size() - 1);
        double currentSignal = signalLine.isEmpty() ? 0 : signalLine.get(signalLine.size() - 1);

        return currentMACD - currentSignal;
    }

    private static List<Double> computeEMASeries(List<Double> data, int period) {
        if (data.size() < period || period <= 0) return new ArrayList<>();

        List<Double> emaSeries = new ArrayList<>();
        double smoothing = 2.0 / (period + 1);

        // Initial SMA calculation
        double sma = data.subList(0, period)
                .stream()
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0);
        emaSeries.add(sma);

        // Wilder's EMA calculation
        for (int i = period; i < data.size(); i++) {
            double prevEMA = emaSeries.get(emaSeries.size() - 1);
            double newEMA = (data.get(i) - prevEMA) * smoothing + prevEMA;
            emaSeries.add(newEMA);
        }

        return emaSeries;
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

    // 4. TRIX Indicator
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

    // 5. Relative Strength Index (RSI)
    public static double calculateRSI(List<StockUnit> prices, int period) {
        if (prices.size() < period + 1) {
            return 0;
        }

        // Calculate initial average gain and loss for the first 'period' days
        double avgGain = 0;
        double avgLoss = 0;

        for (int i = 1; i <= period; i++) {
            double change = prices.get(i).getClose() - prices.get(i - 1).getClose();
            if (change > 0) {
                avgGain += change;
            } else {
                avgLoss += Math.abs(change);
            }
        }

        avgGain /= period;
        avgLoss /= period;

        // Calculate subsequent averages using Wilder's smoothing method
        for (int i = period + 1; i < prices.size(); i++) {
            double change = prices.get(i).getClose() - prices.get(i - 1).getClose();
            double gain = (change > 0) ? change : 0;
            double loss = (change < 0) ? Math.abs(change) : 0;

            avgGain = (avgGain * (period - 1) + gain) / period;
            avgLoss = (avgLoss * (period - 1) + loss) / period;
        }

        if (avgLoss == 0) {
            return 100;
        }

        double rs = avgGain / avgLoss;
        return 100 - (100 / (1 + rs));
    }

    // 6. Rate of Change (ROC) with SIMD optimization
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

    // 7. Momentum Oscillator with look back optimization
    public static double calculateMomentum(List<StockUnit> window, int periods) {
        if (window.size() < periods + 1) return 0;

        double currentClose = window.get(window.size() - 1).getClose();
        double pastClose = window.get(window.size() - 1 - periods).getClose();
        return currentClose - pastClose;
    }

    // 8. Chande Momentum Oscillator (CMO)
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

    // 9. Bollinger Bands with Bandwidth Expansion
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

    // 10. Consecutive Positive Closes with Momentum Tolerance
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

    // 11. Higher Highs Pattern with Adaptive Window
    public static int isHigherHighs(List<StockUnit> window, int minConsecutive) {
        if (window.size() < minConsecutive) return 0;

        for (int i = window.size() - minConsecutive; i < window.size() - 1; i++) {
            if (window.get(i + 1).getClose() <= window.get(i).getClose()) {
                return 0;
            }
        }
        return 1;
    }

    // 12. Automated Trend-line Analysis
    public static int isTrendLineBreakout(List<StockUnit> window, int lookBack) {
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

    // 13. Cumulative Percentage Change with threshold check
    public static int isCumulativeSpike(List<StockUnit> window, int period, double threshold) {
        if (window.size() < period) return 0;

        double sum = window.stream()
                .skip(window.size() - period)
                .mapToDouble(StockUnit::getPercentageChange)
                .sum();

        // Return 1 if true, 0 if false
        return sum >= threshold ? 1 : 0;
    }

    // 14. Cumulative Percentage Change
    private static double cumulativePercentageChange(List<StockUnit> stocks) {
        int startIndex = 0;
        try {
            startIndex = stocks.size() - 5;
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Using parallelStream to process the list in parallel
        return stocks.subList(startIndex, stocks.size())
                .stream()
                .mapToDouble(StockUnit::getPercentageChange)  // Convert to double
                .sum();  // Sum all the results
    }

    // 15. Parabolic SAR Approximation using Close Prices
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

    // 16. Keltner Channels Breakout
    public static int isKeltnerBreakout(List<StockUnit> window, int emaPeriod, int atrPeriod, double multiplier) {
        double ema = calculateEMA(window, emaPeriod);
        double atr = calculateATR(window, atrPeriod);
        double upperBand = ema + (multiplier * atr);

        // Return 1 if the close is greater than the upper band, otherwise return 0
        return (window.get(window.size() - 1).getClose() > upperBand) ? 1 : 0;
    }

    // 17. Elder-Ray Index Approximation
    public static double elderRayIndex(List<StockUnit> window, int emaPeriod) {
        double ema = calculateEMA(window, emaPeriod);
        return window.get(window.size() - 1).getClose() - ema;
    }

    // 18. ATR Calculator using Close Prices (adjusted from traditional)
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
        alertsList.add(new Notification(String.format("%.3f%% %s ↑ %s", totalChange, symbol, prediction), String.format("Increased by %.3f%% at the %s", totalChange, date.format(DateTimeFormatter.ofPattern("dd-MM-yyyy HH:mm:ss"))), timeSeries, date, symbol, totalChange));
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