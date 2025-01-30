package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static org.crecker.Main_UI.addNotification;
import static org.crecker.Main_data_handler.*;
import static org.crecker.data_tester.*;

public class P_L_Tester {
    public static void main(String[] args) {
        //updateStocks();
        PLAnalysis();
    }

    private static void updateStocks() {
        for (String stock : Arrays.asList("SMCI", "IONQ", "WOLF", "MARA")) {
            getData(stock);
        }
    }

    public static void PLAnalysis() {
        // Configuration - use symbol names without .txt extension
        final String[] SYMBOLS = {"MARA.txt", "IONQ.txt", "SMCI.txt", "WOLF.txt"};
        final double INITIAL_CAPITAL = 130000;
        final int FEE = 0;
        final int stockNumber = 0;
        final double dipLevel = -0.5;
        final int revPCall = 10;

        prepData(SYMBOLS, 20000);
        System.out.println("Data prep done");

        double capital = INITIAL_CAPITAL;
        int successfulCalls = 0;
        LocalDateTime lastTradeTime = null;

        System.out.println(notificationsForPLAnalysis.size() + " notification entries");

        for (Notification notification : notificationsForPLAnalysis) {
            try {
                String stockSymbol = notification.getSymbol();

                // Get from symbol timeline instead of stockList
                StockUnit currentUnit = findInSymbolTimeline(stockSymbol, notification.getLocalDateTime());

                if (currentUnit == null) {
                    System.out.println("Error: Time is not in TimeLine");
                    continue;
                }

                if (shouldProcessDip(currentUnit, dipLevel, lastTradeTime)) {
                    TradeResult result = processSymbolTradeSequence(currentUnit, dipLevel, capital);
                    capital = result.newCapital() - FEE;
                    successfulCalls++;
                    lastTradeTime = result.lastTradeTime();
                    //createNotification(notification);
                    logTradeResult(stockSymbol, result);
                    //getNext5Minutes(capital, result.lastTradeTime(), stockSymbol);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        logFinalResults(dipLevel, capital, INITIAL_CAPITAL, successfulCalls, revPCall);

        //createTimeline(SYMBOLS[stockNumber]);
    }

    private static StockUnit findInSymbolTimeline(String symbol, LocalDateTime timestamp) {
        List<StockUnit> timeline = getSymbolTimeline(symbol);

        return timeline.stream()
                .filter(unit -> unit.getLocalDateTimeDate().equals(timestamp))
                .findFirst()
                .orElse(null);
    }

    private static boolean shouldProcessDip(StockUnit unit, double dipLevel, LocalDateTime lastTradeTime) {
        return unit.getPercentageChange() <= dipLevel && (lastTradeTime == null || unit.getLocalDateTimeDate().isAfter(lastTradeTime));
    }

    private static TradeResult processSymbolTradeSequence(StockUnit startUnit, double dipLevel, double capital) {
        List<StockUnit> timeline = symbolTimelines.get(startUnit.getSymbol());
        if (timeline == null) return new TradeResult(capital, startUnit.getLocalDateTimeDate());

        int startIndex = timeline.indexOf(startUnit);
        if (startIndex == -1) return new TradeResult(capital, startUnit.getLocalDateTimeDate());

        double currentCapital = capital;
        LocalDateTime currentTime = startUnit.getLocalDateTimeDate();

        for (int i = startIndex + 1; i < timeline.size(); i++) {
            StockUnit nextUnit = timeline.get(i);
            currentCapital *= (1 + (nextUnit.getPercentageChange() / 100));
            currentTime = nextUnit.getLocalDateTimeDate();

            if (nextUnit.getPercentageChange() > dipLevel) {
                break;
            }
        }

        return new TradeResult(currentCapital, currentTime);
    }

    private static void logTradeResult(String symbol, TradeResult result) {
        System.out.printf("%s trade: Final capital %.2f at %s%n",
                symbol,
                result.newCapital(),
                result.lastTradeTime().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
        );
    }

    private static void logFinalResults(double dipLevel, double capital, double initial, int calls, int revPCall) {
        double revenue = (capital - initial) * 0.75;
        if (calls > 0 && revenue / calls >= revPCall) {
            System.out.printf("Dip Level: %.2f%n", dipLevel);
            System.out.printf("Total Revenue: €%.2f%n", revenue);
            System.out.printf("Successful Calls: %d%n", calls);
            System.out.printf("Revenue/Call: €%.2f%n%n", revenue / calls);
        }
    }

    private static void prepData(String[] fileNames, int cut) {
        // Calculation of spikes, Process data for each file
        for (String fileName : fileNames) {
            try {
                processStockDataFromFile(fileName, fileName.substring(0, fileName.indexOf(".")), cut);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        data_tester.calculateStockPercentageChange();
        spikeDetector();
    }

    public static void processStockDataFromFile(String filePath, String symbol, int retainLast) throws IOException {
        List<StockUnit> fileUnits = readStockUnitsFromFile(filePath, retainLast);
        symbol = symbol.toUpperCase();

        List<StockUnit> existing = symbolTimelines.getOrDefault(symbol, new ArrayList<>());
        existing.addAll(fileUnits);
        symbolTimelines.put(symbol, existing);

        System.out.println("Loaded " + fileUnits.size() + " entries for " + symbol);
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
                    System.out.println("Failed to parse entry: " + entry + " | Error: " + e.getMessage());
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
                    currentEvent.getTimeSeries(), currentEvent.getColor(),
                    currentEvent.getLocalDateTime(), currentEvent.getSymbol(),
                    currentEvent.getChange());
        } catch (Exception ignored) {
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
            System.err.println("Error creating timeline: " + e.getMessage());
        }
    }

    private static void getNext5Minutes(double capital, LocalDateTime startTime, String symbol) {
        symbol = symbol.toUpperCase();
        List<StockUnit> timeline = symbolTimelines.getOrDefault(symbol, Collections.emptyList());

        if (timeline.isEmpty()) {
            System.out.println("No data available for " + symbol);
            return;
        }

        // Find starting index
        int startIndex = -1;
        for (int i = 0; i < timeline.size(); i++) {
            if (timeline.get(i).getLocalDateTimeDate().equals(startTime)) {
                startIndex = i;
                break;
            }
        }

        if (startIndex == -1) {
            System.out.println("Start time not found in data: " + startTime);
            return;
        }

        System.out.printf("\u001B[34mSimulating %s from %s with $%.2f\u001B[0m%n",
                symbol, startTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME), capital);

        double simulatedCapital = capital;
        int predictionsMade = 0;

        for (int i = 1; i <= 5 && (startIndex + i) < timeline.size(); i++) {
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