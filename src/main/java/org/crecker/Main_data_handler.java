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
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.DayOfWeek;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.stream.Collectors;

import static org.crecker.Main_UI.logTextArea;

public class Main_data_handler {
    public static ArrayList<stock> stockList = new ArrayList<>();
    public static int frameSize = 20; // Frame size for analysis
    public static int timeWindow = 1; //time window between alerts
    public static List<Notification> notificationsForPLAnalysis = new ArrayList<>();

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

                    if (response != null) {
                        data[4] = response.getPERatio();
                        data[5] = response.getPEGRatio();
                        data[6] = response.getFiftyTwoWeekHigh();
                        data[7] = response.getFiftyTwoWeekLow();
                        data[8] = Double.valueOf(response.getMarketCapitalization());
                    } else {
                        System.out.println("Company overview response is null.");
                    }
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();

        AlphaVantage.api()
                .timeSeries()
                .quote()
                .forSymbol(symbol_name)
                .onSuccess(e -> {
                    QuoteResponse response = (QuoteResponse) e;

                    if (response != null) {
                        data[0] = response.getOpen();
                        data[1] = response.getHigh();
                        data[2] = response.getLow();
                        data[3] = response.getVolume();
                    } else {
                        System.out.println("Quote response is null.");
                    }

                    // Call the callback with the fetched data
                    callback.onDataFetched(data);
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static Date convertToDate(String timestamp) {
        // Convert timestamp to Date
        try {
            return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(timestamp);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            return new Date();
        }
    }

    public static void handleFailure(AlphaVantageException error) {
        System.out.println("error: " + error.getMessage());
    }

    public static void findMatchingSymbols(String searchText, SymbolSearchCallback callback) {
        List<String> allSymbols = new ArrayList<>();

        AlphaVantage.api()
                .Stocks()
                .setKeywords(searchText)
                .onSuccess(e -> {
                    List<StockResponse.StockMatch> list = e.getMatches();

                    for (StockResponse.StockMatch stockResponse : list) {
                        allSymbols.add(stockResponse.getSymbol());
                    }
                    // Filter and invoke the callback on success
                    List<String> filteredSymbols = allSymbols.stream()
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
                    System.out.println(e.getMessage());
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
                        System.out.println(e.getMessage());
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
                                    long volume = ((TimeSeriesResponse) tsResponse).getStockUnits().get(0).getVolume();

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

        while (true) {
            try {
                List<RealTimeResponse.RealTimeMatch> matches = new ArrayList<>();
                // CountDownLatch to wait for all API calls to finish
                CountDownLatch latch = new CountDownLatch((int) Math.ceil(symbols.size() / 100.0));

                for (int i = 0; i < Math.ceil(symbols.size() / 100.0); i++) {
                    String symbolsBatch = String.join(",", symbols.subList(i * 100, Math.min((i + 1) * 100, symbols.size()))).toUpperCase();
                    AlphaVantage.api()
                            .Realtime()
                            .setSymbols(symbolsBatch)
                            .onSuccess(response -> {
                                matches.addAll(response.getMatches());
                                latch.countDown(); // Decrement the latch count when a batch completes
                            })
                            .onFailure(Main_data_handler::handleFailure)
                            .fetch();
                }

                // Wait for all async calls to complete
                latch.await(); // This will block until all counts down to 0

                // Once all async calls are completed, process the matches
                processStockData(matches);

                // Wait for 5 seconds before repeating the function
                Thread.sleep(5000);

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt(); // Restore interrupted status
                logTextArea.append("Error occurred during data pull\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
                break; // Exit the loop if interrupted
            }
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
        // Create a new timeframe of StockUnit objects
        List<StockUnit> stockBatch = new ArrayList<>();

        // Build StockUnit objects for each RealTimeMatch
        for (RealTimeResponse.RealTimeMatch match : matches) {
            StockUnit stockUnit = new StockUnit.Builder()
                    .symbol(match.getSymbol())
                    .close(match.getClose())
                    .time(match.getTimestamp())
                    .build();

            stockBatch.add(stockUnit);
        }

        // Create a stock object for the current batch
        stock stockObj = new stock(new ArrayList<>(stockBatch));

        // Apply the same logic used for matchList
        if (stockList.isEmpty() || stockList.size() == 1) {
            stockList.add(stockObj); // Add the stock object to stockList
        } else {
            // Compare the size of the current stock batch with the previous one
            if (stockList.get(stockList.size() - 1).stockUnits.size() == matches.size()) {
                stockList.add(stockObj); // Add the stock object to stockList
            } else {
                System.out.println("stockSize doesn't match: " + stockList.get(stockList.size() - 1).stockUnits.size() + " vs. " + matches.size());
            }
        }

        calculateStockPercentageChange();

        logTextArea.append("New stock data got processed\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
    }

    public static void calculateStockPercentageChange() {
        // Check if there are at least two batches of stock data in stockList
        if (stockList.size() > 1) {
            // Get the last two batches
            stock currentStockBatch = stockList.get(stockList.size() - 1);
            stock previousStockBatch = stockList.get(stockList.size() - 2);

            // Check if the current batch and previous batch have the same number of stock units
            if (currentStockBatch.stockUnits.size() == previousStockBatch.stockUnits.size()) {
                // Iterate through the stock units in the current and previous batches
                for (int i = 0; i < currentStockBatch.stockUnits.size(); i++) {
                    // Get the current and previous stock units
                    StockUnit currentStockUnit = currentStockBatch.stockUnits.get(i);
                    StockUnit previousStockUnit = previousStockBatch.stockUnits.get(i);

                    // Get the current close price and the previous close price
                    double currentClose = currentStockUnit.getClose();
                    double previousClose = previousStockUnit.getClose();

                    // Calculate the percentage change between the consecutive stock units
                    double percentageChange = ((currentClose - previousClose) / previousClose) * 100;

                    // Check for a 14% dip or peak
                    if (Math.abs(percentageChange) >= 14) {
                        currentStockUnit.setPercentageChange(previousStockUnit.getPercentageChange());

                    } else {
                        // Set the percentage change using the setter method
                        currentStockUnit.setPercentageChange(percentageChange);
                    }
                }

                logTextArea.append("Stock percentage changes calculated and updated for the latest two batches.\n");
            } else {
                logTextArea.append("The number of stock units in the current and previous batches do not match.\n");
            }
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        } else {
            logTextArea.append("Not enough stock data available to calculate percentage change.\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        }

        calculateSpikes();
    }

    public static void calculateSpikes() {
        spikeDetector();
        checkToClean();
    }

    public static void checkToClean() {
        // Get the Java runtime
        Runtime runtime = Runtime.getRuntime();

        // Calculate the used memory
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();  // In bytes
        long usedMemoryInMB = usedMemory / (1024 * 1024);  // Convert bytes to MB

        // Print the current used memory for debugging
        System.out.println("Used memory: " + usedMemoryInMB + " MB");

        // Check if the used memory exceeds 400 MB
        if (usedMemoryInMB > 500) {
            stockList.subList(0, 200).clear();
            System.out.println("stockList cleared due to excessive memory usage");
        }
    }

    /**
     * Detects potential spikes in stock data and generates notifications based on the analysis.
     * It processes frames of stock data, filters notifications, and sorts them for analysis.
     */
    public static void spikeDetector() {
        if (stockList.size() > frameSize && stockList.size() > 4) { //check if frame is in size
            for (int k = 0; k < stockList.get(frameSize - 1).stockUnits.size(); k++) { //go through all symbols
                getFullFrame(k);
                //getRealFrame(k);
            }
        }

        timeWindow = 0;
        // Before sort all notifications which are significant included
        // After sort as well but mixed with other notifications
        filterNotificationsByTimeWindow(notificationsForPLAnalysis);
    }

    /**
     * Processes a single frame of stock data to generate real-time notifications.
     *
     * @param k Index of the stock symbol in the frame.
     */
    private static void getRealFrame(int k) {
        List<StockUnit> frame = new ArrayList<>();

        try {
            // Create a frame of the last `frameSize` stock units
            for (int j = stockList.size() - 1 - frameSize; j < stockList.size() - 1; j++) {
                frame.add(stockList.get(j).stockUnits.get(k));
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

        // Get notifications for the current frame
        List<Notification> notifications = getNotificationForFrame(frame, stockList.get(stockList.size() - 1).stockUnits.get(k).getSymbol());

        if (!notifications.isEmpty()) {
            notificationsForPLAnalysis.addAll(notifications);
        }
    } //Build method for real data


    /**
     * Processes multiple frames of stock data to generate notifications for all frames.
     *
     * @param k Index of the stock symbol in the frame.
     */
    public static void getFullFrame(int k) {
        List<Notification> stockNotification = new ArrayList<>();

        for (int i = frameSize + 1; i < stockList.size() - 1; i++) {
            List<StockUnit> frame = new ArrayList<>();
            try {
                // Create a frame of the last `frameSize` stock units, rolling over each iteration
                for (int j = i - frameSize; j < i; j++) {
                    frame.add(stockList.get(j).stockUnits.get(k));
                }
            } catch (Exception e) {
                continue;
            }

            // Get notifications for the current frame
            List<Notification> notifications = getNotificationForFrame(frame, stockList.get(4).stockUnits.get(k).getSymbol());

            if (!notifications.isEmpty()) { // Emptiness check
                stockNotification.addAll(notifications);
            }
        }

        filterNotificationsByTimeWindow(stockNotification);
        notificationsForPLAnalysis.addAll(stockNotification);
    }

    /**
     * Filters notifications to ensure they are within a defined time window and removes duplicates.
     *
     * @param notifications List of notifications to filter.
     */
    public static void filterNotificationsByTimeWindow(List<Notification> notifications) {
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

        // Use a LinkedHashSet to maintain insertion order and ensure uniqueness
        Set<Notification> filteredNotifications = new LinkedHashSet<>();
        LocalDateTime lastNotificationTime = null;

        for (Notification notification : notifications) {
            LocalDateTime currentNotificationTime = parseNotificationTime(notification);

            if (currentNotificationTime != null) {
                // Check uniqueness and time window
                if (lastNotificationTime == null || isOutsideTimeWindow(currentNotificationTime, lastNotificationTime)) {
                    filteredNotifications.add(notification);
                    lastNotificationTime = currentNotificationTime; // Update last notification time
                }
            } else {
                // Log or track invalid notifications for debugging
                System.err.println("Invalid notification format: " + notification);
            }
        }

        // Replace the original list with filtered notifications
        notifications.clear();
        notifications.addAll(filteredNotifications);
    }

    /**
     * Checks if a given notification timestamp is outside the specified time window.
     *
     * @param current The timestamp of the current notification.
     * @param last    The timestamp of the last valid notification.
     * @return True if the current notification is outside the time window; false otherwise.
     */
    private static boolean isOutsideTimeWindow(LocalDateTime current, LocalDateTime last) {
        return current.isAfter(last.plusMinutes(timeWindow));
    }

    /**
     * Parses the timestamp from the content of a notification.
     *
     * @param notification The notification whose timestamp needs to be parsed.
     * @return The parsed timestamp as a LocalDateTime object, or null if parsing fails.
     */
    private static LocalDateTime parseNotificationTime(Notification notification) {
        int atIndex = notification.getContent().indexOf("at the ");
        if (atIndex == -1) {
            return null; // Invalid format
        }

        String datePart = notification.getContent().substring(atIndex + "at the ".length()).trim();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        try {
            return LocalDateTime.parse(datePart, formatter);
        } catch (Exception e) {
            return null; // Failed to parse
        }
    }

    /**
     * Generates notifications based on patterns and criteria within a frame of stock data.
     *
     * @param stocks    The frame of stock data.
     * @param stockName The name of the stock being analyzed.
     * @return A list of notifications generated from the frame.
     */
    public static List<Notification> getNotificationForFrame(List<StockUnit> stocks, String stockName) {
        //prevent wrong dip variables (optimized)
        double lastChanges = 0;
        int lastChangeLength = 5;

        //minor dip detection variables (optimized)
        int consecutiveIncreaseCount = 0;
        double cumulativeIncrease = 0;
        double cumulativeDecrease = 0;
        double minorDipTolerance = 0.65;

        //rapid increase variables (optimized)
        double minIncrease = 0.25;
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
        TimeSeries timeSeries = new TimeSeries(stockName);

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
                timeSeries.add(new Minute(Main_data_handler.convertToDate(stocks.get(i).getDate())), stocks.get(i).getClose());
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
                        createNotification(stockName, lastChanges, alertsList, timeSeries, stocks.get(i).getDate(), true);
                    } catch (Exception ignored) {
                    }
                }

                //rapid increase logic
                rapidIncreaseLogic(stocks, stockName, i, rapidWindowSize,
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
        // Parse the first and last stock unit's dates
        LocalDateTime startDate = LocalDateTime.parse(stocks.get(0).getDate().replace(' ', 'T'));
        LocalDateTime endDate = LocalDateTime.parse(stocks.get(stocks.size() - 1).getDate().replace(' ', 'T'));

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
            double maxIncreaseInWindow = 0.0;

            for (int j = i - rapidWindowSize + 1; j <= i; j++) {
                double increase = stocks.get(j).getPercentageChange();
                maxIncreaseInWindow = Math.max(maxIncreaseInWindow, increase);
            }

            if (((volatility >= volatilityThreshold) || (lastChanges > 1.2)) &&
                    ((maxIncreaseInWindow >= minIncrease)) &&
                    (consecutiveIncreaseCount >= minConsecutiveCount) &&
                    (i >= (stocks.size() - lastChangeLength)) &&
                    (lastChanges > minIncrease)) {

                createNotification(stockName, maxIncreaseInWindow, alertsList, timeSeries, stocks.get(i).getDate(), false);
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
     * @param stockName   The name of the stock.
     * @param totalChange The total percentage change triggering the notification.
     * @param alertsList  The list to store the notification.
     * @param timeSeries  The time series for graphical representation.
     * @param date        The date of the event.
     * @param dip         True if the event is a dip; false if it's an increase.
     */
    private static void createNotification(String stockName, double totalChange, List<Notification> alertsList, TimeSeries timeSeries, String date, boolean dip) {
        if ((totalChange > 0) && !dip) {
            alertsList.add(new Notification(String.format("%.3f%% %s stock increase", totalChange, stockName), String.format("Increased by %.3f%% at the %s", totalChange, date), timeSeries, new Color(50, 205, 50)));
        } else if (dip) {
            alertsList.add(new Notification(String.format("%.3f%% %s stock dipped", totalChange, stockName), String.format("dipped by %.3f%% at the %s", totalChange, date), timeSeries, new Color(255, 217, 0)));
        }
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

    // OOP Models - reduced to one -> average 25% less RAM used
    public static class stock {
        ArrayList<StockUnit> stockUnits;

        public stock(ArrayList<StockUnit> stockUnits) {
            this.stockUnits = stockUnits;
        }
    }
}