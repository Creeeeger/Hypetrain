package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jetbrains.annotations.NotNull;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.*;

import static org.crecker.Main_UI.logTextArea;

public class data_tester {
    static List<StockUnit> inter_day_stocks;
    static List<Notification> alerts;

    public static void main(String[] args) throws IOException {
        List<StockUnit> stocks = readStockUnitsFromFile("TSLA.txt"); //Get the Stock data from the file (simulate real Stock data)

        for (Notification notification : tester(stocks, true, 1)) {
            System.out.println(notification.getTitle() + " " + notification.getContent());
        }
    }

    public static List<StockUnit> readStockUnitsFromFile(String filePath) throws IOException {
        // Read the entire file content as a single string
        BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath));
        StringBuilder fileContentBuilder = new StringBuilder();
        String line;

        while ((line = bufferedReader.readLine()) != null) {
            fileContentBuilder.append(line).append("\n");
        }

        String fileContent = fileContentBuilder.toString().trim();

        // Close the reader as we're done reading the file
        bufferedReader.close();

        // Trim the leading '[' and trailing ']' if present
        if (fileContent.startsWith("[")) {
            fileContent = fileContent.substring(1).trim();
        }

        if (fileContent.endsWith("]")) {
            fileContent = fileContent.substring(0, fileContent.length() - 1).trim();
        }

        String[] stockUnitStrings = fileContent.split("}, ");

        // Initialize the list to hold StockUnit objects
        List<StockUnit> stockUnits = new ArrayList<>();

        // Iterate over each StockUnit string and parse it
        for (String stockUnitString : stockUnitStrings) {
            // Clean up any trailing curly braces and whitespaces
            stockUnitString = stockUnitString.trim();
            if (stockUnitString.endsWith("}")) {
                stockUnitString = stockUnitString.substring(0, stockUnitString.length() - 1);
            }

            // Parse the string and convert it to a StockUnit object
            StockUnit stockUnit = parseStockUnit(stockUnitString);
            stockUnits.add(stockUnit);
        }

        // Reverse the list to get the Stock units in chronological order since the dumb ass api gives us the stuff in the wrong direction
        Collections.reverse(stockUnits);

        return stockUnits;
    }

    private static StockUnit parseStockUnit(String stockUnitString) {
        // Remove "StockUnit{" from the beginning of the string
        stockUnitString = stockUnitString.replace("StockUnit{", "").trim();

        // Split the Stock unit attributes by commas
        String[] attributes = stockUnitString.split(", ");

        // Parse each attribute
        double open = Double.parseDouble(attributes[0].split("=")[1]);
        double high = Double.parseDouble(attributes[1].split("=")[1]);
        double low = Double.parseDouble(attributes[2].split("=")[1]);
        double close = Double.parseDouble(attributes[3].split("=")[1]);
        double adjustedClose = Double.parseDouble(attributes[4].split("=")[1]);
        long volume = Long.parseLong(attributes[5].split("=")[1]);
        double dividendAmount = Double.parseDouble(attributes[6].split("=")[1]);
        double splitCoefficient = Double.parseDouble(attributes[7].split("=")[1]);
        String dateTime = attributes[8].split("=")[1];

        // Use the Builder to create the StockUnit object
        return new StockUnit.Builder()
                .open(open)
                .high(high)
                .low(low)
                .close(close)
                .adjustedClose(adjustedClose)
                .volume(volume)
                .dividendAmount(dividendAmount)
                .splitCoefficient(splitCoefficient)
                .time(dateTime)
                .build();
    }

    public static List<Notification> tester(List<StockUnit> stocks, Boolean allTime, int time) {
        inter_day_stocks = get_Inter_Day(stocks, Main_data_handler.convertToDate_Simple(stocks.get(time).getDate()), allTime);

        alerts = get_alerts_from_stock(inter_day_stocks);

        Stock_value(inter_day_stocks);

        return alerts;
    }

    public static List<StockUnit> get_Inter_Day(List<StockUnit> stocks, Date last_date, Boolean allTime) {
        List<StockUnit> inter_day_stocks = new ArrayList<>();

        for (int i = 0; i < stocks.size(); i++) {
            Date current_date = Main_data_handler.convertToDate_Simple(stocks.get(i).getDate());

            // Check if the current date matches the last date
            if (current_date.equals(last_date) || allTime) {
                double current_close = stocks.get(i).getClose();

                // Ensure there is a previous Stock entry to compare with
                if (i > 0) {
                    double previous_close = stocks.get(i - 1).getClose();

                    // Check for a 10% dip or peak
                    if (Math.abs((current_close - previous_close) / previous_close) >= 0.1) {
                        // Replace the current close with the previous close
                        stocks.get(i).setClose(previous_close); // Use the setter method
                    }
                }

                // Add the modified Stock to the inter_day_stocks list
                inter_day_stocks.add(stocks.get(i));
            }
        }

        return inter_day_stocks;
    }

    public static List<Notification> get_alerts_from_stock(List<StockUnit> stocks) {
        List<Notification> alertsList = new ArrayList<>();
        Set<String> uniqueAlerts = new HashSet<>(); // To track unique alerts
        int frameSize = 20;

        for (int i = frameSize; i < stocks.size(); i++) {
            List<StockUnit> frame = new ArrayList<>();

            for (int j = 0; j < frameSize; j++) {
                frame.add(stocks.get(i - frameSize + j));
            }

            // Get notifications for the current frame
            List<Notification> notifications = getNotificationForFrame(frame, "Nvidia");

            // Add unique notifications to the alertsList
            if (!notifications.isEmpty()) { // Emptiness check
                for (Notification notification : notifications) {
                    // Check if the alert is already added
                    if (uniqueAlerts.add(notification.toString())) { // Add to the Set and check for uniqueness
                        alertsList.add(notification); // Add to the alerts list if unique
                    }
                }
            }
        }

        return alertsList;
    }

    public static List<Notification> getNotificationForFrame(List<StockUnit> stocks, String stockName) {
        //Crash Related variables
        int minCrashLevelPercentage = 5;
        int crashLength = 10;

        //prevent wrong dip variables
        double lastChanges = 0;
        int lastChangeLength = 5;

        //permanent rise variables
        double permanentChangeLevel = 4.0;

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
        double totalChange = 0;
        String pattern = "";

        for (int i = 1; i < stocks.size(); i++) {
            //Changes & percentages calculations
            double currentClose = stocks.get(i).getClose();
            double previousClose = stocks.get(i - 1).getClose();
            double percentageChange = ((currentClose - previousClose) / previousClose) * 100;
            percentageChanges.add(percentageChange);
            timeSeries.add(new Minute(Main_data_handler.convertToDate(stocks.get(i).getDate())), stocks.get(i).getClose());

            //Crash logic
            crashLogic(stocks, stockName, i, crashLength, minCrashLevelPercentage, alertsList, timeSeries);

            //last changes calculations
            lastChanges = LastChangeLogic(stocks, i, lastChangeLength, lastChanges, percentageChange);

            //total change calculation
            totalChange = TotalChangeLogic(stocks, stockName, totalChange, percentageChange, i, permanentChangeLevel, alertsList, timeSeries);

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
                double previousPrice = stocks.get(j - 1).getClose();
                double currentPrice = stocks.get(j).getClose();
                double increase = ((currentPrice - previousPrice) / previousPrice) * 100;

                maxIncreaseInWindow = Math.max(maxIncreaseInWindow, increase);
            }

            if ((volatility >= volatilityThreshold) && (maxIncreaseInWindow >= minIncrease) && (consecutiveIncreaseCount >= minConsecutiveCount) && (i >= (stocks.size() - lastChangeLength)) && (lastChanges > minIncrease)) {
                Set<String> undesiredPatterns = getStrings();
                // Avoid undesired patterns
                if (!undesiredPatterns.contains(pattern)) {
                    createNotification(stockName, maxIncreaseInWindow, alertsList, timeSeries, false, stocks.get(i).getDate());
                }
            }
        }
    }

    private static double TotalChangeLogic(List<StockUnit> stocks, String stockName, double totalChange, double percentageChange, int i, double permanentChangeLevel, List<Notification> alertsList, TimeSeries timeSeries) {
        totalChange += percentageChange;
        if (i == stocks.size() - 1) {
            if (totalChange > permanentChangeLevel) {
                createNotification(stockName, totalChange, alertsList, timeSeries, true, stocks.get(stocks.size() - 1).getDate());
            }
        }
        return totalChange;
    }

    private static double LastChangeLogic(List<StockUnit> stocks, int i, int lastChangeLength, double lastChanges, double percentageChange) {
        if (i >= (stocks.size() - lastChangeLength)) {
            lastChanges += percentageChange;
        }
        return lastChanges;
    }

    private static void crashLogic(List<StockUnit> stocks, String stockName, int i, int crashLength, int minCrashLevelPercentage, List<Notification> alertsList, TimeSeries timeSeries) {
        double maxPriceInWindow = Double.MIN_VALUE;
        double minPriceInWindow = Double.MAX_VALUE;

        if (i > crashLength) {
            for (int j = i - crashLength; j <= i; j++) {
                double price = stocks.get(j).getClose();
                maxPriceInWindow = Math.max(maxPriceInWindow, price);
                minPriceInWindow = Math.min(minPriceInWindow, price);
            }

            double crash = ((maxPriceInWindow - minPriceInWindow) / maxPriceInWindow) * 100;

            // Check for a crash of 5% or more in the window
            if (crash >= minCrashLevelPercentage) {
                crash = crash * -1;
                createNotification(stockName, crash, alertsList, timeSeries, false, stocks.get(i).getDate());
            }
        }
    }

    // Helper method for calculating volatility (standard deviation of percentage changes)
    private static double calculateVolatility(List<Double> changes) {
        double mean = changes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = changes.stream().mapToDouble(change -> Math.pow(change - mean, 2)).average().orElse(0.0);
        return Math.sqrt(variance); // Standard deviation as volatility measure
    }

    private static void createNotification(String stockName, double totalChange, List<Notification> alertsList, TimeSeries timeSeries, boolean longTimeIncrease, String date) {
        if (longTimeIncrease) {
            alertsList.add(new Notification(String.format("%.3f%% %s stock steady rise", totalChange, stockName), String.format("rose by %.3f%% at the %s", totalChange, date), timeSeries, new Color(50, 200, 150)));
        } else if (totalChange > 0) {
            alertsList.add(new Notification(String.format("%.3f%% %s stock increase", totalChange, stockName), String.format("Increased by %.3f%% at the %s", totalChange, date), timeSeries, new Color(50, 205, 50)));
        } else {
            alertsList.add(new Notification(String.format("%.3f%% %s stock decrease", totalChange, stockName), String.format("Decreased by %.3f%% at the %s", totalChange, date), timeSeries, new Color(178, 34, 34)));
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

    public static void Stock_value(List<StockUnit> stocks) { //plot the stock value
        // Create a TimeSeries object for plotting
        TimeSeries timeSeries = new TimeSeries("NVDA Stock Price");

        // Populate the time series with Stock data
        for (StockUnit stock : stocks) {
            String timestamp = stock.getDate();
            double closingPrice = stock.getClose(); // Assuming getClose() returns closing price

            // Add the data to the TimeSeries
            timeSeries.add(new Minute(Main_data_handler.convertToDate(timestamp)), closingPrice);
        }

        // Plot the data
        Main_data_handler.plotData(timeSeries, "NVDA price change", "Date", "price");
    }

    public static List<Notification> Main_data_puller() throws IOException { //get stock notifications
        logTextArea.append("Data puller has started.\n");

        List<StockUnit> stocks = readStockUnitsFromFile("TSLA.txt"); //Get the Stock data from the file (simulate real Stock data)

        logTextArea.append("Data puller has finished.\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        return tester(stocks, false, 5); //test method to test the Stock data
    }
}