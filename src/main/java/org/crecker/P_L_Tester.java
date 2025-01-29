package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
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
        //variables
        double capital;  //capital variable
        double dipLevel;
        int fee = 0;
        int stock = 3;
        int calls;
        LocalDateTime lastClose;
        double capitalOriginal;  //capital variable
        double revenue; //revenue counter
        String[] fileNames = {"MARA.txt", "IONQ.txt", "SMCI.txt", "WOLF.txt"}; //add more files

        prepData(fileNames);

        for (double i = 1; i > -1; i -= 0.05) {
            //Reset variables
            capital = 130000;
            dipLevel = i;
            calls = 0;
            lastClose = null;
            capitalOriginal = capital;

            //looping through the notifications
            for (Notification currentEvent : notificationsForPLAnalysis) {
                // Extract notification date and format it
                Date notificationDate = currentEvent.getTimeSeries()
                        .getTimePeriod(currentEvent.getTimeSeries().getItemCount() - 1)
                        .getEnd();

                Calendar calendar = Calendar.getInstance();
                calendar.setTime(notificationDate);
                calendar.set(Calendar.SECOND, 0); // Set seconds to 00
                String outputDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(calendar.getTime());

                // Extract stock name
                String stockName = currentEvent.getTitle()
                        .substring(currentEvent.getTitle().indexOf("%") + 2, currentEvent.getTitle().indexOf(" stock"));

                // Find the stock in the array
                int stockInArray = -1;
                for (int j = 0; j < stockList.get(4).getStockUnits().size(); j++) {
                    if (Objects.equals(stockList.get(4).getStockUnits().get(j).getSymbol(), stockName)) {
                        stockInArray = j;
                        break;
                    }
                }

                // Handle case when stock is not found
                if (stockInArray == -1) {
                    System.out.println("Stock not found: " + stockName);
                    continue;
                }

                // Find the matching date in the array
                int dateInArray = -1;
                for (int j = 4; j < stockList.size(); j++) {
                    if (Objects.equals(stockList.get(j).getStockUnits().get(stockInArray).getDate(), outputDate)) {
                        dateInArray = j;
                        break;
                    }
                }

                // Next dip method
                if (!(stockList.get(dateInArray + 1).getStockUnits().get(stockInArray).getPercentageChange() <= dipLevel)) {

                    if (lastClose != null && lastClose.isAfter(LocalDateTime.parse(stockList.get(dateInArray + 1).getStockUnits().get(stockInArray).getDate(), DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")))) {
                        continue;
                    }

                    calls++;

                    while (dateInArray < stockList.size() - 2) {  // Ensure we don't go out of bounds

                        double percentageChange = 0;
                        try {
                            percentageChange = stockList.get(dateInArray + 2).getStockUnits().get(stockInArray).getPercentageChange();
                        } catch (Exception ignored) {
                        }

                        // Update capital based on the percentage change
                        capital = capital * (1 + (percentageChange / 100));

                        // If the percentage change is above the dip level, continue processing
                        if (percentageChange <= dipLevel) {
                            lastClose = LocalDateTime.parse(stockList.get(dateInArray + 2).getStockUnits().get(stockInArray).getDate(), DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
                            break;  // Exit the loop if the percentage change is no longer above dipLevel
                        }

                        // Move to the next data point
                        dateInArray++;
                    }

                    getNext5Minutes(capital, dateInArray, fileNames[stockInArray].toUpperCase().replace(".TXT", ""));

                    //createNotification(currentEvent);

                    // Subtract trading fee
                    capital = capital - fee;
                }
            }

            revenue = (capital - capitalOriginal) * 0.75;
            if (!(revenue / calls < 400)) {
                System.out.printf("%.2f%n", i);
                System.out.printf("Total Revenue %s€\n", String.format("%.2f", revenue)); // print out results
                System.out.println("Calls:" + calls);
                System.out.printf("Revenue per call: %.2f \n", revenue / calls);
                System.out.println();
            }
        }

        // createTimeline(fileNames, stock);
    }

    private static void prepData(String[] fileNames) {
        // Calculation of spikes, Process data for each file
        for (String fileName : fileNames) {
            try {
                processStockDataFromFile(fileName, fileName.substring(0, fileName.indexOf(".")), 30000);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        data_tester.calculateStockPercentageChange();
        spikeDetector();
    }

    public static void processStockDataFromFile(String filePath, String symbol, int retainLast) throws IOException {
        List<StockUnit> units = readStockUnitsFromFile(filePath, retainLast);

        stockList.addAll(units.stream()
                .peek(u -> u.setSymbol(symbol)) // Ensure correct symbol
                .map(u -> {
                    Map<String, StockUnit> batch = new HashMap<>();
                    batch.put(symbol, u);
                    return new stock(batch);
                })
                .collect(Collectors.toList()));
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
            addNotification(currentEvent.getTitle(), currentEvent.getContent(), currentEvent.getTimeSeries(), currentEvent.getColor());
        } catch (Exception ignored) {
        }
    }

    private static void createTimeline(String[] fileNames, int stockIndex) {
        try {
            String symbol = fileNames[stockIndex].toUpperCase().replace(".TXT", "");

            TimeSeries timeSeries = new TimeSeries(symbol + " Timeline");

            for (int i = 1; i < stockList.size(); i++) {
                StockUnit currentUnit = stockList.get(i).getStockUnits().get(symbol);

                if (currentUnit != null) {
                    double closePrice = currentUnit.getClose();

                    // Add to time series
                    timeSeries.addOrUpdate(
                            new Minute(convertToDate(currentUnit.getDate())),
                            closePrice
                    );
                }
            }

            plotData(timeSeries, symbol + " Price Trend", "Date", "Price");
        } catch (Exception e) {
            System.err.println("Failed to create timeline: " + e.getMessage());
        }
    }

    private static void getNext5Minutes(double capital, int startIndex, String symbol) {
        if (stockList.size() < startIndex + 6) {
            System.out.println("Not enough data for simulation");
            return;
        }

        double simulatedCapital = capital;
        symbol = symbol.toUpperCase();

        System.out.printf("\u001B[34mStarting simulation for %s with $%.2f\u001B[0m%n", symbol, capital);

        for (int i = 1; i <= 5; i++) {
            int targetIndex = startIndex + i + 1;
            if (targetIndex >= stockList.size()) break;

            StockUnit futureUnit = stockList.get(targetIndex).getStockUnits().get(symbol);
            if (futureUnit == null) {
                System.out.println("Missing future data at index " + targetIndex);
                continue;
            }

            double change = futureUnit.getPercentageChange();
            simulatedCapital *= (1 + (change / 100));

            System.out.printf(
                    "\u001B[33mEvent %d: %+.2f%% on %s → New Value: $%.2f\u001B[0m%n",
                    i, change, futureUnit.getDate(), simulatedCapital
            );
        }

        System.out.printf(
                "\u001B[32mFinal simulated value: $%.2f (%.2f%% change)\u001B[0m%n",
                simulatedCapital,
                ((simulatedCapital - capital) / capital) * 100
        );
    }
}