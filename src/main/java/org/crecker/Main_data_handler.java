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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static org.crecker.Main_UI.logTextArea;

public class Main_data_handler {
    public static Map<String, List<StockUnit>> symbolTimelines = new HashMap<>();
    public static int frameSize = 20; // Frame size for analysis
    public static List<Notification> notificationsForPLAnalysis = new ArrayList<>();
    public static boolean test = false; //if True use demo url for real Time Updates

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

        calculateSpikes();
    }

    public static void calculateSpikes() {
        spikeDetector();
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

    /**
     * Detects potential spikes in stock data and generates notifications based on the analysis.
     * It processes frames of stock data, filters notifications, and sorts them for analysis.
     */
    public static void spikeDetector() {
        symbolTimelines.keySet()
                .parallelStream()
                .forEach(symbol -> {
                    List<StockUnit> timeline = symbolTimelines.get(symbol);
                    if (timeline != null && timeline.size() > frameSize) {
                        getFrame(symbol);
                    }
                });

        sortNotifications(notificationsForPLAnalysis);
    }

    /**
     * Processes multiple frames for a specific symbol to generate notifications
     *
     * @param symbol The stock symbol to analyze
     */
    public static void getFrame(String symbol) {
        List<Notification> stockNotification = new ArrayList<>();
        List<StockUnit> timeline = getSymbolTimeline(symbol);

        if (timeline.size() <= frameSize) {
            System.out.println("TimeLine to small");
            return;
        }

        // Slide window through timeline
        for (int i = frameSize; i < timeline.size(); i++) {
            List<StockUnit> frame = timeline.subList(i - frameSize, i);

            try {
                List<Notification> notifications = getNotificationForFrame(
                        frame,
                        symbol // Use actual symbol from timeline
                );

                if (!notifications.isEmpty()) {
                    stockNotification.addAll(notifications);
                }
            } catch (Exception e) {
                e.printStackTrace();
                System.out.print("Frame processing failed for " + symbol + " at index " + i + "\n");
            }
        }

        synchronized (notificationsForPLAnalysis) {
            notificationsForPLAnalysis.addAll(stockNotification);
        }
    }

    public static void sortNotifications(List<Notification> notifications) {
        // Sort notifications by their time series end date
        notifications.sort((n1, n2) -> {
            Date date1 = n1.getTimeSeries()
                    .getTimePeriod(n1.getTimeSeries().getItemCount() - 1)
                    .getEnd();
            Date date2 = n2.getTimeSeries()
                    .getTimePeriod(n2.getTimeSeries().getItemCount() - 1)
                    .getEnd();
            return date1.compareTo(date2); // Sort from old to new
        });
    }

    /**
     * Generates notifications based on patterns and criteria within a frame of stock data.
     *
     * @param stocks The frame of stock data.
     * @param symbol The name of the stock being analyzed.
     * @return A list of notifications generated from the frame.
     */
    public static List<Notification> getNotificationForFrame(List<StockUnit> stocks, String symbol) {
        //prevent wrong dip variables (optimized)
        double lastChanges = 0;
        int lastChangeLength = 5;

        //minor dip detection variables (optimized)
        int consecutiveIncreaseCount = 0;
        double cumulativeIncrease = 0;
        double cumulativeDecrease = 0;
        double minorDipTolerance = 0.65;

        //rapid increase variables (optimized)
        double minIncrease = 2;
        int rapidWindowSize = 1;
        int minConsecutiveCount = 2;

        //Volatility variables (optimized)
        double volatility;
        double volatilityThreshold = 0.07;

        //Crash variables (optimized)
        double dipDown = -1.5;
        double dipUp = 0.8;
        // Target is to minimize the crashes and maximize
        // the profit or keep the ratio balanced since
        // we don't want any extremes of both

        //algorithm related variables
        List<Notification> alertsList = new ArrayList<>();
        List<Double> percentageChanges = new ArrayList<>();
        TimeSeries timeSeries = new TimeSeries(symbol);

        /*
        Algorithm parts:
        - isWeekendSpan: check if frame is over weekend
        - percentageChange: gets the percentage change in numerical format for percentage as percent times by 100
        - timeSeries: adds the frame to the timeSeries for later presentation
        - lastChanges: sum of all percentages over the frame selected
        - cumulativeIncrease / cumulativeDecrease: check if increase an increase and allow small dips, check as well for consecutive increases
        - volatility: calculate the volatility of the frame
        - rapidIncreaseLogic: check if frame satisfies criteria for notification
         */

        // Prevent notifications if time frame spans over the weekend (Friday to Monday)
        if (isWeekendSpan(stocks)) {
            return new ArrayList<>(); // Return empty list to skip notifications
        }

        for (int i = 1; i < stocks.size(); i++) {
            //Changes & percentages calculations
            double percentageChange = stocks.get(i).getPercentageChange();
            percentageChanges.add(percentageChange);

            try {
                timeSeries.add(new Minute(stocks.get(i).getDateDate()), stocks.get(i).getClose());
            } catch (Exception ignored) {
            }

            //last changes calculations
            lastChanges = LastChangeLogic(stocks, i, lastChangeLength, lastChanges, percentageChange);

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
                volatility = calculateVolatility(percentageChanges);

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
                        volatility, volatilityThreshold, minIncrease, consecutiveIncreaseCount,
                        minConsecutiveCount, lastChangeLength, lastChanges, alertsList, timeSeries);
            }
        }
        return alertsList;
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
     * Implements logic to detect rapid increases in stock prices within a specified frame.
     *
     * @param stocks                   The frame of stock data.
     * @param stockName                The name of the stock being analyzed.
     * @param i                        The current index within the frame.
     * @param rapidWindowSize          The size of the rapid increase window.
     * @param volatility               The volatility of the frame.
     * @param volatilityThreshold      The threshold for volatility.
     * @param minIncrease              The minimum percentage increase to qualify.
     * @param consecutiveIncreaseCount The count of consecutive increases.
     * @param minConsecutiveCount      The minimum consecutive increase count required.
     * @param lastChangeLength         The length of the change tracking window.
     * @param lastChanges              The cumulative percentage change in the window.
     * @param alertsList               The list to store generated notifications.
     * @param timeSeries               The time series for graphical representation.
     */
    private static void rapidIncreaseLogic(List<StockUnit> stocks, String stockName, int i,
                                           int rapidWindowSize, double volatility, double volatilityThreshold,
                                           double minIncrease, int consecutiveIncreaseCount, int minConsecutiveCount,
                                           int lastChangeLength, double lastChanges, List<Notification> alertsList, TimeSeries timeSeries) {
        if (i >= rapidWindowSize) { // Ensure the window is valid
            if (((volatility >= volatilityThreshold) || (lastChanges > 1.2)) &&
                    (consecutiveIncreaseCount >= minConsecutiveCount) &&
                    (i >= (stocks.size() - lastChangeLength)) &&
                    (lastChanges > minIncrease)) {

                createNotification(stockName, lastChanges, alertsList, timeSeries, stocks.get(i).getLocalDateTimeDate(), false);

                //       System.out.printf("Name: %s Volatility %.2f vs %.2f Consecutive %s vs %s, Last Change %.2f vs %.2f Date %s%n", stockName, volatility, volatilityThreshold, consecutiveIncreaseCount, minConsecutiveCount, lastChanges, minIncrease, stocks.get(i).getStringDate());
            }
        }
    }

    /**
     * Calculates cumulative percentage changes for a stock within a specified length.
     *
     * @param stocks           The frame of stock data.
     * @param i                The current index within the frame.
     * @param lastChangeLength The length of the change tracking window.
     * @param lastChanges      The cumulative percentage change.
     * @param percentageChange The current percentage change.
     * @return Updated cumulative percentage change.
     */
    private static double LastChangeLogic(List<StockUnit> stocks, int i, int lastChangeLength, double lastChanges, double percentageChange) {
        if (i >= (stocks.size() - lastChangeLength)) {
            lastChanges += percentageChange;
        }
        return lastChanges;
    }

    /**
     * Calculates the volatility (standard deviation) of percentage changes in stock data.
     *
     * @param changes A list of percentage changes in stock prices.
     * @return The calculated volatility.
     */
    private static double calculateVolatility(List<Double> changes) {
        double mean = changes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = changes.stream().mapToDouble(change -> Math.pow(change - mean, 2)).average().orElse(0.0);
        return Math.sqrt(variance); // Standard deviation as volatility measure
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
}