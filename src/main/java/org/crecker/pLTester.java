package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static org.crecker.Main_UI.addNotification;
import static org.crecker.Main_UI.gui;
import static org.crecker.Main_data_handler.*;
import static org.crecker.data_tester.*;
import static org.crecker.data_tester.getData;

public class pLTester {
    // Index map for quick timestamp lookups
    private static final Map<String, Map<LocalDateTime, Integer>> symbolTimeIndex = new ConcurrentHashMap<>();
    public static boolean debug = false; // Flag for printing PL

    public static void main(String[] args) {
        //updateStocks();
        long startTime = System.nanoTime(); // Record the start time

        PLAnalysis(); // Call the method you want to monitor

        long endTime = System.nanoTime(); // Record the end time
        long durationInNanoseconds = endTime - startTime;

        System.out.println("Execution time: " + (durationInNanoseconds / 1_000_000) + " milliseconds");
    }

    private static void updateStocks() {
        for (String stock : Arrays.asList("SMCI", "IONQ", "WOLF", "MARA")) {
            getData(stock);
        }
    }

    public static void PLAnalysis() {
        //   final String[] SYMBOLS = {"MARA.txt", "IONQ.txt", "SMCI.txt", "WOLF.txt"};
        final String[] SYMBOLS = {"SMCI.txt"};

        final double INITIAL_CAPITAL = 130000;
        final int FEE = 0;
        final int stock = 0;
        final double DIP_LEVEL = -0.5;

        prepData(SYMBOLS, 800);

        // Preprocess indices during data loading
        Arrays.stream(SYMBOLS).forEach(symbol -> buildTimeIndex(symbol.replace(".txt", ""),
                getSymbolTimeline(symbol.replace(".txt", ""))));

        double capital = INITIAL_CAPITAL;
        int successfulCalls = 0;
        LocalDateTime lastTradeTime = null;

        // Cache timelines per notification
        Map<String, List<StockUnit>> timelineCache = new HashMap<>();

        for (Notification notification : notificationsForPLAnalysis) {
            if (gui != null) {
                createNotification(notification);
            }

            String symbol = notification.getSymbol();
            List<StockUnit> timeline = timelineCache.computeIfAbsent(symbol, Main_data_handler::getSymbolTimeline);

            Integer index = getIndexForTime(symbol, notification.getLocalDateTime());

            if (index == null || index >= timeline.size() - 1) {
                System.out.println("Invalid time index for " + symbol);
                continue;
            }

            StockUnit nextUnit = timeline.get(index + 1);

            if (shouldProcessDip(nextUnit, DIP_LEVEL, lastTradeTime)) {
                TradeResult result = processTradeSequence(timeline, index + 2, DIP_LEVEL, capital, notification.getSymbol());
                capital = result.newCapital() - FEE;
                successfulCalls++;
                lastTradeTime = result.lastTradeTime();
                if (debug) {
                    logTradeResult(symbol, result);
                    getNext5Minutes(capital, lastTradeTime, notification.getSymbol());
                }
            }
        }
        createTimeline(SYMBOLS[stock]);
        if (debug) {
            createTimeline(SYMBOLS[stock]);
            logFinalResults(DIP_LEVEL, capital, INITIAL_CAPITAL, successfulCalls);
        }
    }

    private static void buildTimeIndex(String symbol, List<StockUnit> timeline) {
        Map<LocalDateTime, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < timeline.size(); i++) {
            indexMap.put(timeline.get(i).getLocalDateTimeDate(), i);
        }
        symbolTimeIndex.put(symbol, indexMap);
    }

    private static Integer getIndexForTime(String symbol, LocalDateTime time) {
        return symbolTimeIndex.getOrDefault(symbol, Collections.emptyMap()).get(time);
    }

    private static boolean shouldProcessDip(StockUnit nextUnit, double dipLevel, LocalDateTime lastTradeTime) {
        return nextUnit.getPercentageChange() >= dipLevel && (lastTradeTime == null || nextUnit.getLocalDateTimeDate().isAfter(lastTradeTime));
    }

    private static TradeResult processTradeSequence(List<StockUnit> timeline, int startIndex, double dipLevel, double capital, String symbol) {
        double currentCapital = capital;
        int currentIndex = startIndex;
        final int maxSteps = Math.min(timeline.size(), startIndex + 100); // Safety limit

        while (currentIndex < maxSteps) {
            StockUnit unit = timeline.get(currentIndex);
            currentCapital *= (1 + (unit.getPercentageChange() / 100));

            if (debug) {
                System.out.printf("%s trade: capital %.2f change %.2f Date: %s%n", symbol, capital, unit.getPercentageChange(), unit.getDateDate());
            }

            if (unit.getPercentageChange() < dipLevel) {
                break;
            }
            currentIndex++;
        }

        // catch errors in case not enough data points are available
        try {
            return new TradeResult(currentCapital, timeline.get(currentIndex).getLocalDateTimeDate());
        } catch (Exception e) {
            return new TradeResult(currentCapital, timeline.get(currentIndex - 1).getLocalDateTimeDate());
        }
    }

    private static void logTradeResult(String symbol, TradeResult result) {
        System.out.printf("%s trade: Final capital %.2f at %s%n",
                symbol,
                result.newCapital(),
                result.lastTradeTime().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
        );
    }

    private static void logFinalResults(double dipLevel, double capital, double initial, int calls) {
        double revenue = (capital - initial) * 0.75;
        if (calls > 0) {
            System.out.printf("Dip Level: %.2f%n", dipLevel);
            System.out.printf("Total Revenue: €%.2f%n", revenue);
            System.out.printf("Successful Calls: %d%n", calls);
            System.out.printf("Revenue/Call: €%.2f%n%n", revenue / calls);
        }
    }

    private static void prepData(String[] fileNames, int cut) {
        // Calculation of rallies, Process data for each file
        Arrays.stream(fileNames).parallel().forEach(fileName -> {
            try {
                processStockDataFromFile(fileName, fileName.substring(0, fileName.indexOf(".")), cut);
            } catch (IOException e) {
                e.printStackTrace();
            }
        });

        data_tester.calculateStockPercentageChange();
        rallyDetector(frameSize, false);
    }

    public static void processStockDataFromFile(String filePath, String symbol, int retainLast) throws IOException {
        List<StockUnit> fileUnits = readStockUnitsFromFile(filePath, retainLast);
        symbol = symbol.toUpperCase();

        List<StockUnit> existing = symbolTimelines.getOrDefault(symbol, new ArrayList<>());
        existing.addAll(fileUnits);
        symbolTimelines.put(symbol, existing);

        System.out.println("Loaded " + fileUnits.size() + " entries for " + symbol);

        if (debug) {
            exportToCSV(fileUnits);
        }
    }

    public static void exportToCSV(List<StockUnit> stocks) {
        try {
            // Create a new FileWriter, overwriting the existing file
            FileWriter csvWriter = new FileWriter(System.getProperty("user.dir") + "/rallyMLModel/highFrequencyStocks.csv");
            // Write the CSV header
            csvWriter.append("timestamp,open,high,low,close,volume\n");

            // Define the date format for timestamps
            SimpleDateFormat dateFormat = new SimpleDateFormat("EEE MMM dd HH:mm:ss zzz yyyy");
            // Regex pattern to validate the timestamp format (e.g., "Fri Jan 03 14:30:00 GMT 2025")
            String timestampPattern = "[A-Za-z]{3} [A-Za-z]{3} \\d{2} \\d{2}:\\d{2}:\\d{2} [A-Za-z]{3} \\d{4}";

            // Iterate over each StockUnit
            for (StockUnit stock : stocks) {

                // Step 2: Format and validate the timestamp
                try {
                    String timestamp = dateFormat.format(stock.getDateDate());
                    // Step 3: Check if the timestamp matches the expected format
                    if (!timestamp.matches(timestampPattern)) {
                        System.err.println("Warning: Invalid timestamp format: " + timestamp);
                        continue; // Skip this line
                    }

                    // If all checks pass, write the line to the CSV
                    csvWriter.append(escapeCSV(timestamp)).append(",")
                            .append(escapeCSV(String.valueOf(stock.getOpen()))).append(",")
                            .append(escapeCSV(String.valueOf(stock.getHigh()))).append(",")
                            .append(escapeCSV(String.valueOf(stock.getLow()))).append(",")
                            .append(escapeCSV(String.valueOf(stock.getClose()))).append(",")
                            .append(escapeCSV(String.valueOf(stock.getVolume()))).append("\n");
                } catch (Exception e) {
                    // Catch any formatting errors (e.g., if getDateDate() returns an invalid type)
                    System.err.println("Warning: Failed to format timestamp for stock unit");
                    continue; // Skip this line
                }
            }

            // Flush and close the writer
            csvWriter.flush();
            csvWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static String escapeCSV(String data) {
        if (data == null) {
            return "";  // Handle null values by returning an empty string
        }

        String escapedData = data; // Initialize escapedData with the original data

        // If data contains commas, quotes, or newlines, enclose it in double quotes
        if (data.contains(",") || data.contains("\"") || data.contains("\n")) {
            escapedData = "\"" + data.replace("\"", "\"\"") + "\""; // Replace quotes with escaped quotes and enclose in double quotes
        }
        return escapedData; // Return the escaped data
    }

    public static List<StockUnit> readStockUnitsFromFile(String filePath, int retainLast) throws IOException {
        List<StockUnit> stockUnits = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String content = reader.lines()
                    .collect(Collectors.joining("\n"))
                    .replaceAll("^\\[|]$", "") // Remove array brackets
                    .trim();

            if (content.isEmpty()) {
                System.out.println("The file is empty or incorrectly formatted.");
                return stockUnits;
            }

            // Use the original working split pattern
            String[] entries = content.split("},\\s*");

            for (String entry : entries) {
                try {
                    entry = entry.trim();

                    // Clean up entry format as in original working version
                    if (entry.endsWith("}")) {
                        entry = entry.substring(0, entry.length() - 1);
                    }

                    // Handle potential nested braces from StockUnit class
                    entry = entry.replace("StockUnit{", "").trim();

                    stockUnits.add(parseStockUnit(entry));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        // Maintain original reversal and retention logic
        Collections.reverse(stockUnits);
        int keepFrom;
        try {
            keepFrom = Math.max(0, stockUnits.size() - retainLast);
        } catch (Exception e) {
            keepFrom = 0;
        }
        return new ArrayList<>(stockUnits.subList(keepFrom, stockUnits.size()));
    }

    private static void createNotification(Notification currentEvent) {
        try {
            addNotification(currentEvent.getTitle(), currentEvent.getContent(),
                    currentEvent.getTimeSeries(), currentEvent.getLocalDateTime(),
                    currentEvent.getSymbol(), currentEvent.getChange());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void createTimeline(String symbol) {
        try {
            symbol = symbol.toUpperCase().replace(".TXT", "");
            List<StockUnit> timeline = getSymbolTimeline(symbol);

            if (timeline.isEmpty()) {
                System.out.println("No data available for " + symbol);
                return;
            }

            TimeSeries timeSeries = new TimeSeries(symbol + " Price Timeline");

            for (StockUnit unit : timeline) {
                timeSeries.addOrUpdate(
                        new Minute(unit.getDateDate()),
                        unit.getClose()
                );
            }

            plotData(timeSeries, symbol + " Historical Prices", "Time", "Price");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void getNext5Minutes(double capital, LocalDateTime startTime, String symbol) {
        symbol = symbol.toUpperCase();
        List<StockUnit> timeline = symbolTimelines.getOrDefault(symbol, Collections.emptyList());

        if (timeline.isEmpty()) {
            System.out.println("No data available for " + symbol);
            return;
        }

        Integer startIndex = symbolTimeIndex.getOrDefault(symbol, Collections.emptyMap()).get(startTime);

        if (startIndex == null || startIndex < 0 || startIndex >= timeline.size()) {
            System.out.println("Start time not found in data: " + startTime);
            return;
        }

        System.out.printf("\u001B[34mSimulating %s from %s with $%.2f\u001B[0m%n",
                symbol, startTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME), capital);

        double simulatedCapital = capital;
        int predictionsMade = 0;
        final int maxSteps = Math.min(5, timeline.size() - startIndex - 1);

        for (int i = 1; i <= maxSteps; i++) {
            StockUnit futureUnit = timeline.get(startIndex + i);
            double change = futureUnit.getPercentageChange();
            simulatedCapital *= (1 + (change / 100));
            predictionsMade++;

            System.out.printf("\u001B[33m%d min later: %+.2f%% on %s → $%.2f\u001B[0m%n",
                    i, change,
                    futureUnit.getLocalDateTimeDate().format(DateTimeFormatter.ISO_LOCAL_TIME),
                    simulatedCapital
            );
        }

        System.out.printf("\u001B[32mFinal simulation result: $%.2f (%.2f%% change)\u001B[0m%n",
                simulatedCapital,
                ((simulatedCapital - capital) / capital) * 100
        );

        if (predictionsMade < 5) {
            System.out.println("Warning: Only " + predictionsMade + " predictions available");
        }
    }

    // Record for trade results
    private record TradeResult(double newCapital, LocalDateTime lastTradeTime) {
    }
}