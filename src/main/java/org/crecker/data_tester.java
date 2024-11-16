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
        List<StockUnit> stocks = readStockUnitsFromFile("NVDA.txt"); //Get the Stock data from the file (simulate real Stock data)

        tester(stocks, false); //test method to test the Stock data
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

    public static List<Notification> tester(List<StockUnit> stocks, Boolean allTime) {
        inter_day_stocks = get_Inter_Day(stocks, Main_data_handler.convertToDate_Simple(stocks.get(5000).getDate()), allTime);

        alerts = get_alerts_from_stock(inter_day_stocks);

        Stock_value(inter_day_stocks);
        Stock_smoothed_change(inter_day_stocks, 5);

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

        for (int i = 20; i < stocks.size(); i++) {
            List<StockUnit> frame = new ArrayList<>();

            // Add 20 stocks to the frame in the correct order
            for (int j = 0; j < 20; j++) {
                frame.add(stocks.get(i - 20 + j));
            }

            // Get notifications for the current frame
            List<Notification> notifications = getNotificationForFrame(frame, "Nvidia");

            // Add notifications to alertsList if not empty
            if (!notifications.isEmpty()) {
                alertsList.addAll(notifications); // Add all notifications to alertsList
            }
        }

        return alertsList;
    }

    //!!!Finish percentage algorithm
    public static List<Notification> getNotificationForFrame(List<StockUnit> stocks, String stockName) {
        List<Notification> alertsList = new ArrayList<>();

        // Thresholds for early detection
        double volatilityThreshold = 0.05;
        double cumulativeChangeThreshold = 1.0;
        double cumulativeChange = 0;
        double change15to20 = 0;

        List<Double> percentageChanges = new ArrayList<>();
        TimeSeries timeSeries = new TimeSeries(stockName + " stock chart");

        // Initialize consecutive positive change counters to track momentum
        int consecutiveIncreaseCount = 0;
        double cumulativeIncrease = 0;
        double cumulativeDecrease = 0;
        double minorDipTolerance = 0.2; // Allow dips up to 20% of cumulative increase

        for (int i = 1; i < stocks.size(); i++) {
            double currentClose = stocks.get(i).getClose();
            double previousClose = stocks.get(i - 1).getClose();
            double percentageChange = ((currentClose - previousClose) / previousClose) * 100;

            timeSeries.add(new Minute(Main_data_handler.convertToDate(stocks.get(i).getDate())), stocks.get(i).getClose());
            percentageChanges.add(percentageChange);

            if (i >= 15) {
                change15to20 += percentageChange;
            }

            if (i >= 10) {
                cumulativeChange += percentageChange;
            }

            // Check if the current percentage change is positive or a minor dip
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
        }

        String pattern = detectPattern(percentageChanges);

        // Calculate volatility as the standard deviation of percentage changes
        double volatility = calculateVolatility(percentageChanges);

        // Apply predictive spike detection logic
        if (isPredictiveSpikeEvent(change15to20, cumulativeChange, volatility, volatilityThreshold, cumulativeChangeThreshold, consecutiveIncreaseCount, pattern)) {
            if (cumulativeChange > 0) {
                alertsList.add(new Notification(cumulativeChange + "% " + stockName + " stock predicted increase ", String.format("%s, %.3f", cumulativeChange, volatility), timeSeries, new Color(50, 205, 50)));
            } else {
                alertsList.add(new Notification(cumulativeChange + "% " + stockName + " stock predicted decrease ", String.format("%s, %.3f", cumulativeChange, volatility), timeSeries, new Color(178, 34, 34)));
            }
            System.out.println(cumulativeChange + "% " + stockName + " stock predicted decrease " + String.format("%s, %.3f", cumulativeChange, volatility));
        }

        return alertsList;
    }

    // Predictive spike detection method focusing on early rise indicators
    private static boolean isPredictiveSpikeEvent(double change17to20, double cumulativeChange, double volatility,
                                                  double volatilityThreshold, double cumulativeChangeThreshold,
                                                  int consecutiveIncreaseCount, String pattern) {
        // Set of undesired patterns to avoid
        Set<String> undesiredPatterns = getStrings();

        // Avoid undesired patterns
        if (undesiredPatterns.contains(pattern)) {
            return false;
        } else {
            // Early spike detection criteria:
            return (                            // At least one period with positive change for early detection
                    (volatility >= volatilityThreshold)               // Ensure enough volatility
                            && (cumulativeChange >= cumulativeChangeThreshold || consecutiveIncreaseCount >= 3)
                            && (change17to20 > 0)
            ); // Accumulated change or sustained momentum
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

    // Helper method for calculating volatility (standard deviation of percentage changes)
    private static double calculateVolatility(List<Double> changes) {
        double mean = changes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = changes.stream().mapToDouble(change -> Math.pow(change - mean, 2)).average().orElse(0.0);
        return Math.sqrt(variance); // Standard deviation as volatility measure
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

    public static void Stock_smoothed_change(List<StockUnit> stocks, int windowSize) {
        // Create a TimeSeries object for plotting
        TimeSeries timeSeries = new TimeSeries("NVDA Smoothed Percentage Change");

        List<Double> percentageChanges = new ArrayList<>();

        // Calculate percentage changes and store them
        for (int i = 1; i < stocks.size(); i++) {
            double currentClose = stocks.get(i).getClose();  // Get the current close price
            double previousClose = stocks.get(i - 1).getClose();  // Get the previous close price

            // Calculate percentage change
            double percentageChange = ((currentClose - previousClose) / previousClose) * 100;
            percentageChanges.add(percentageChange);
        }

        // Apply the moving average to smooth the data
        for (int i = windowSize - 1; i < percentageChanges.size(); i++) {
            String date = stocks.get(i).getDate();

            double sum = 0.0;
            for (int j = i - windowSize + 1; j <= i; j++) {
                sum += percentageChanges.get(j);
            }

            // Calculate the moving average
            double smoothedChange = sum / windowSize;

            // Add the smoothed value to the TimeSeries
            timeSeries.add(new Minute(Main_data_handler.convertToDate(date)), smoothedChange);
        }

        // Plot the data
        Main_data_handler.plotData(timeSeries, "NVDA Smoothed Percentage Change", "Date", "Smoothed Change (%)");
    }

    public static List<Notification> Main_data_puller() throws IOException { //get stock notifications
        logTextArea.append("Data puller has started.\n");

        List<StockUnit> stocks = readStockUnitsFromFile("NVDA.txt"); //Get the Stock data from the file (simulate real Stock data)

        logTextArea.append("Data puller has finished.\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        return tester(stocks, false); //test method to test the Stock data
    }
}

//TODO
//!!!Finish percentage algorithm