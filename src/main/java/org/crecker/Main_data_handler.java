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
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;

import java.awt.*;
import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.stream.Collectors;

import static org.crecker.Main_UI.addNotification;
import static org.crecker.Main_UI.logTextArea;

public class Main_data_handler {
    public static ArrayList<stock> stockList = new ArrayList<>();
    public static int frameSize = 20; // Frame size for analysis

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
        String[] stockSymbols = { //List of theoretical trade-able symbols
                "AEM", "CVNA", "VRT", "ODFL", "EW", "KHC", "VLO", "CTVA", "HES", "TCOM",
                "MCHP", "LNG", "CBRE", "GLW", "FERG", "ACGL", "SNOW", "IT", "LVS", "DDOG",
                "LULU", "AME", "DFS", "EA", "GIS", "YUM", "IRM", "VRSK", "IDXX", "HSY",
                "BKR", "SYY", "NGG", "BNS", "NXPI", "COF", "EPD", "WDAY", "RSG", "AJG",
                "FTNT", "ADSK", "AFL", "PCG", "PSA", "DHI", "TTD", "SLB", "MET", "ROP",
                "GM", "SE", "TSN", "MKC", "EXPE", "TRU", "MDB", "MT", "ZTO", "BBY",
                "ARE", "OMC", "LPLA", "AER", "CLX", "PFG", "ULTA", "VRSN", "LUV", "DGX",
                "APTV", "DKNG", "UTHR", "AKAM", "FTAI", "POOL", "MGM", "CAVA", "TTEK", "SWK",
                "GGG", "DUOL", "BMRN", "XPO", "AES", "TXRH", "DOCU", "GMAB", "CF", "USFD",
                "REG", "SWKS", "TFC", "TRV", "NSC", "ET", "OKE", "APP", "URI", "PSX",
                "GWW", "O", "COIN", "AZO", "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
                "TSM", "BRK/B", "TSLA", "LLY", "AVGO", "WMT", "JPM", "XOM", "UNH", "V",
                "NVO", "ORCL", "MA", "HD", "PG", "COST", "JNJ", "ABBV", "BAC", "NFLX",
                "KO", "CRM", "SAP", "ASML", "CVX", "MRK", "TMUS", "AMD", "BABA", "SMFG",
                "TM", "NVS", "PEP", "AZN", "LIN", "WFC", "CSCO", "ADBE", "MCD", "TMO",
                "PM", "ABT", "IBM", "NOW", "MS", "QCOM", "GE", "AXP", "CAT", "TXN",
                "ISRG", "FMX", "DHR", "RY", "VZ", "DIS", "PDD", "NEE", "AMGN", "INTU",
                "RTX", "HSBC", "GS", "UBER", "PFE", "HDB", "CMCSA", "T", "UL", "ARM",
                "AMAT", "SPGI", "LOW", "TTE", "BLK", "BKNG", "PGR", "UNP", "SNY", "SYK",
                "HON", "LMT", "SCHW", "TJX", "BSX", "KKR", "ANET", "VRTX", "C", "BX",
                "MUFG", "COP", "RACE", "MU", "PANW", "NKE", "ADP", "UPS", "CB", "MDT",
                "ADI", "FI", "BUD", "SBUX", "DE", "UBS", "GILD", "IBN", "MMC", "PLD",
                "SONY", "BMY", "AMT", "REGN", "SHOP", "PLTR", "SO", "LRCX", "INTC", "TD",
                "ELV", "ICE", "BA", "HCA", "MDLZ", "INFY", "SHW", "KLAC", "DUK", "RELX",
                "SCCO", "TT", "PBR", "CI", "EQIX", "ABNB", "ENB", "MO", "PYPL", "MCO",
                "CEG", "CTAS", "BP", "CMG", "GD", "WM", "APH", "RIO", "ZTS", "BN",
                "APO", "CME", "AON", "PH", "GEV", "WELL", "CL", "GSK", "BTI", "MSI",
                "SNPS", "ITW", "SPOT", "USB", "TDG", "NOC", "PNC", "TRI", "DEO", "CRWD",
                "CNQ", "MAR", "EQNR", "ECL", "CP", "CVS", "MRVL", "MMM", "APD", "CNI",
                "TGT", "ORLY", "CDNS", "BDX", "EOG", "BMO", "FCX", "CARR", "FDX", "MCK",
                "JD", "CSX"
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
        hardcoreCrash(20);
        spikeDetector();
        checkToClean();
    }

    public static void hardcoreCrash(int entries) {
        logTextArea.append("Checking for crashes\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

        // Ensure we have enough stock data to compare
        if (stockList.size() >= entries) {
            // Initialize a list to track the cumulative percentage changes for each stock unit (skip stock 0)
            List<Double> changes = new ArrayList<>(Collections.nCopies(stockList.get(0).stockUnits.size(), 0.0));

            // Loop over the stock batches, starting from the (entries)th last batch to compare with the previous batches
            for (int i = stockList.size() - entries + 1; i < stockList.size(); i++) {
                stock currentStockBatch = stockList.get(i);

                // Loop through all the stock units in the current batch
                for (int j = 0; j < currentStockBatch.stockUnits.size()-1; j++) {  // Start from j = 0 (skip stock 0 only if necessary)
                    StockUnit currentStockUnit = currentStockBatch.stockUnits.get(j);

                    // Get the cumulative percentage change for this stock unit
                    double currentPercentageChange = currentStockUnit.getPercentageChange();

                    // Sum up the percentage changes for each stock unit (skip stock 0 if necessary)
                    changes.set(j, changes.get(j) + currentPercentageChange);

                    // Define a crash threshold (you can adjust this value as needed)
                    if (changes.get(j) < -6.0) {
                        // Create a time series for the crashed stock
                        TimeSeries timeSeries = new TimeSeries(currentStockUnit.getSymbol() + " Time Series");

                        // Populate the time series with the stock's date and closing prices
                        for (int k = stockList.size()-entries ; k < stockList.size(); k++) {
                            timeSeries.add(new Minute(Main_data_handler.convertToDate(stockList.get(k).stockUnits.get(j).getDate())), stockList.get(k).stockUnits.get(j).getClose());
                        }

                        // Report the crash with the time series
                        addNotification(String.format("%.3f%% %s Crash", changes.get(j), currentStockUnit.getSymbol()),
                                String.format("Crashed by %.3f%% at %s", changes.get(j), currentStockUnit.getDate()),
                                timeSeries,                                     // Include the time series data for the stock
                                new Color(178, 34, 34)                 // Color for notification (red)
                        );
                    }
                }
            }
        }
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

    public static void spikeDetector() {
        Set<String> uniqueAlerts = new HashSet<>(); // To track unique alerts
        LocalDateTime lastNotificationTime = null; // To store the time of the last printed notification

        if (stockList.size() > frameSize) { //check if frame is in size
            for (int k = 0; k < stockList.get(frameSize - 1).stockUnits.size(); k++) { //go through all symbols
                lastNotificationTime = getFullFrame(k, lastNotificationTime, uniqueAlerts);
                //   lastNotificationTime = getRealFrame(k, lastNotificationTime, uniqueAlerts);
            }
        }
    }

    @Nullable
    private static LocalDateTime getRealFrame(int k, LocalDateTime lastNotificationTime, Set<String> uniqueAlerts) {
        List<StockUnit> frame = new ArrayList<>();

        // Create a frame of the last `frameSize` stock units
        for (int j = stockList.size() - 1 - frameSize; j < stockList.size() - 1; j++) {
            frame.add(stockList.get(j).stockUnits.get(k));
        }

        // Get notifications for the current frame
        List<Notification> notifications = getNotificationForFrame(frame, stockList.get(stockList.size() - 1).stockUnits.get(k).getSymbol());

        if (!notifications.isEmpty()) { // Emptiness check

            // Add unique notifications to the alertsList and add them via addNotification
            for (Notification notification : notifications) {

                // Check if the notification content contains "rose by" (to filter the stock change notifications)
                if (notification.getContent().contains("rose by")) {
                    int atIndex = notification.getContent().indexOf("at the ");
                    String datePart = notification.getContent().substring(atIndex + "at the ".length()).trim();

                    // Parse the notification time
                    DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
                    LocalDateTime notificationTime = LocalDateTime.parse(datePart, formatter);

                    // Check if the notification time is outside the 15-minute span or if it's the first notification
                    if (lastNotificationTime == null || isOutsideTimeWindow(notificationTime, lastNotificationTime)) {
                        if (uniqueAlerts.add(notification.toString())) { // Ensure uniqueness

                            if (lastNotificationTime == null) {
                                lastNotificationTime = notificationTime; // Update lastNotificationTime
                            }

                            addNotification(notification.getTitle(), notification.getContent(), notification.getTimeSeries(), notification.getColor());
                        }
                    }
                } else {
                    // For non-stock change notifications, allow them if they are unique
                    if (uniqueAlerts.add(notification.toString())) { // Ensure uniqueness
                        lastNotificationTime = null;
                        addNotification(notification.getTitle(), notification.getContent(), notification.getTimeSeries(), notification.getColor());
                    }
                }
            }
        }
        return lastNotificationTime;
    }

    @Nullable
    public static LocalDateTime getFullFrame(int k, LocalDateTime lastNotificationTime, Set<String> uniqueAlerts) {
        for (int i = frameSize + 1; i < stockList.size() - 1; i++) {
            List<StockUnit> frame = new ArrayList<>();

            // Create a frame of the last `frameSize` stock units, rolling over each iteration
            for (int j = i - frameSize; j < i; j++) {
                frame.add(stockList.get(j).stockUnits.get(k));
            }

            // Get notifications for the current frame
            List<Notification> notifications = getNotificationForFrame(frame, stockList.get(stockList.size() - 1).stockUnits.get(k).getSymbol());
            if (!notifications.isEmpty()) { // Emptiness check

                // Add unique notifications to the alertsList and add them via addNotification
                for (Notification notification : notifications) {

                    // Check if the notification content contains "rose by" (to filter the stock change notifications)
                    if (notification.getContent().contains("rose by")) {
                        int atIndex = notification.getContent().indexOf("at the ");
                        String datePart = notification.getContent().substring(atIndex + "at the ".length()).trim();

                        // Parse the notification time
                        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
                        LocalDateTime notificationTime = LocalDateTime.parse(datePart, formatter);

                        // Check if the notification time is outside the 15-minute span or if it's the first notification
                        if (lastNotificationTime == null || isOutsideTimeWindow(notificationTime, lastNotificationTime)) {
                            if (uniqueAlerts.add(notification.toString())) { // Ensure uniqueness

                                if (lastNotificationTime == null) {
                                    lastNotificationTime = notificationTime; // Update lastNotificationTime
                                }
                                addNotification(notification.getTitle(), notification.getContent(), notification.getTimeSeries(), notification.getColor());
                            }
                        }
                    } else {
                        // For non-stock change notifications, allow them if they are unique
                        if (uniqueAlerts.add(notification.toString())) { // Ensure uniqueness
                            lastNotificationTime = null;
                            addNotification(notification.getTitle(), notification.getContent(), notification.getTimeSeries(), notification.getColor());
                        }
                    }
                }
            }
        }
        return lastNotificationTime;
    }

    // Helper method to check if the notification is outside the 15-minute window
    private static boolean isOutsideTimeWindow(LocalDateTime current, LocalDateTime last) {
        return current.isAfter(last.plusMinutes(15));
    }

    public static List<Notification> getNotificationForFrame(List<StockUnit> stocks, String stockName) {
        //prevent wrong dip variables
        double lastChanges = 0;
        int lastChangeLength = 5;

        //minor dip detection variables
        int consecutiveIncreaseCount = 0;
        double cumulativeIncrease = 0;
        double cumulativeDecrease = 0;
        double minorDipTolerance = 0.2;

        //rapid increase variables
        double minIncrease = 0.4;
        int rapidWindowSize = 4;
        int minConsecutiveCount = 1;

        //Volatility variables
        double volatility = 0.0;
        double volatilityThreshold = 0.05;

        //algorithm related variables
        List<Notification> alertsList = new ArrayList<>();
        List<Double> percentageChanges = new ArrayList<>();
        TimeSeries timeSeries = new TimeSeries(stockName);
        String pattern = "";

        for (int i = 1; i < stocks.size(); i++) {
            //Changes & percentages calculations
            double percentageChange = stocks.get(i).getPercentageChange();
            percentageChanges.add(percentageChange);
            timeSeries.add(new Minute(Main_data_handler.convertToDate(stocks.get(i).getDate())), stocks.get(i).getClose());

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
                pattern = detectPattern(percentageChanges);
                volatility = calculateVolatility(percentageChanges);
            }

            //rapid increase logic
            rapidIncreaseLogic(stocks, stockName, i, rapidWindowSize, volatility, volatilityThreshold, minIncrease, consecutiveIncreaseCount, minConsecutiveCount, lastChangeLength, lastChanges, pattern, alertsList, timeSeries);
        }

        return alertsList;
    }

    private static void rapidIncreaseLogic(List<StockUnit> stocks, String stockName, int i, int rapidWindowSize, double volatility, double volatilityThreshold, double minIncrease, int consecutiveIncreaseCount, int minConsecutiveCount, int lastChangeLength, double lastChanges, String pattern, List<Notification> alertsList, TimeSeries timeSeries) {
        if (i >= rapidWindowSize) { // Ensure the window is valid
            double maxIncreaseInWindow = 0.0;

            for (int j = i - rapidWindowSize + 1; j <= i; j++) {
                double increase = stocks.get(i).getPercentageChange();
                maxIncreaseInWindow = Math.max(maxIncreaseInWindow, increase);
            }

            if ((volatility >= volatilityThreshold) && (maxIncreaseInWindow >= minIncrease) && (consecutiveIncreaseCount >= minConsecutiveCount) && (i >= (stocks.size() - lastChangeLength)) && (lastChanges > minIncrease)) {
                Set<String> undesiredPatterns = getStrings();

                // Avoid undesired patterns
                if (!undesiredPatterns.contains(pattern)) {
                    createNotification(stockName, maxIncreaseInWindow, alertsList, timeSeries, stocks.get(i).getDate());
                }
            }
        }
    }

    private static double LastChangeLogic(List<StockUnit> stocks, int i, int lastChangeLength, double lastChanges, double percentageChange) {
        if (i >= (stocks.size() - lastChangeLength)) {
            lastChanges += percentageChange;
        }
        return lastChanges;
    }

    // Helper method for calculating volatility (standard deviation of percentage changes)
    private static double calculateVolatility(List<Double> changes) {
        double mean = changes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = changes.stream().mapToDouble(change -> Math.pow(change - mean, 2)).average().orElse(0.0);
        return Math.sqrt(variance); // Standard deviation as volatility measure
    }

    private static void createNotification(String stockName, double totalChange, List<Notification> alertsList, TimeSeries timeSeries, String date) {
        if (totalChange > 0) {
            alertsList.add(new Notification(String.format("%.3f%% %s stock increase", totalChange, stockName), String.format("Increased by %.3f%% at the %s", totalChange, date), timeSeries, new Color(50, 205, 50)));
        }
    }

    public static String detectPattern(List<Double> percentageChanges) {
        int size = percentageChanges.size();
        int segmentSize = size / 3;

        double change0to033 = 0;
        double change033to066 = 0;
        double change066to1 = 0;

        // Sum the first third of the list
        for (int i = 0; i < segmentSize; i++) {
            change0to033 += percentageChanges.get(i);
        }

        // Sum the second third of the list
        for (int i = segmentSize; i < 2 * segmentSize; i++) {
            change033to066 += percentageChanges.get(i);
        }

        // Sum the last third of the list
        for (int i = 2 * segmentSize; i < size; i++) {
            change066to1 += percentageChanges.get(i);
        }

        return String.valueOf(getPatternSymbol(change0to033)) +
                getPatternSymbol(change033to066) +
                getPatternSymbol(change066to1);
    }

    // Helper method to determine the pattern symbol based on the threshold
    private static char getPatternSymbol(double change) {
        if (change > 0.1) {
            return '/';
        } else if (change >= -0.1 && change < 0.1) {
            return '_';
        } else {
            return '\\';
        }
    }

    @NotNull
    private static Set<String> getStrings() {
        Set<String> undesiredPatterns = new HashSet<>();
        undesiredPatterns.add("\\\\\\");
        undesiredPatterns.add("_\\\\");
        undesiredPatterns.add("__\\");
        undesiredPatterns.add("___");
        undesiredPatterns.add("\\__");
        undesiredPatterns.add("\\\\_");
        undesiredPatterns.add("\\_\\");
        undesiredPatterns.add("//\\\\");
        undesiredPatterns.add("//_\\");
        undesiredPatterns.add("_\\");
        undesiredPatterns.add("//__");
        undesiredPatterns.add("//\\_");
        return undesiredPatterns;
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

    // OOP Models - reduced to one -> average 25% less Ram used
    public static class stock {
        ArrayList<StockUnit> stockUnits;

        public stock(ArrayList<StockUnit> stockUnits) {
            this.stockUnits = stockUnits;
        }
    }
}