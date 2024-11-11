package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

public class data_tester {
    static List<StockUnit> inter_day_stocks;
    static List<Notification> alerts;

    public static void main(String[] args) throws IOException {
        List<StockUnit> stocks = readStockUnitsFromFile("NVDA.txt"); //Get the Stock data from the file (simulate real Stock data)

        tester(stocks); //test method to test the Stock data

        //further code to test on data comes here
        System.out.println("Data got loaded successfully!");
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

    public static List<Notification> tester(List<StockUnit> stocks) {
        inter_day_stocks = get_Inter_Day(stocks, Main_data_handler.convertToDate_Simple(stocks.get(1000).getDate()));

        alerts = get_alerts_from_stock(inter_day_stocks);

        Stock_value(inter_day_stocks);
        //Stock_change(inter_day_stocks);

        return alerts;
    }

    public static List<StockUnit> get_Inter_Day(List<StockUnit> stocks, Date last_date) {
        List<StockUnit> inter_day_stocks = new ArrayList<>();

        for (int i = 0; i < stocks.size(); i++) {
            Date current_date = Main_data_handler.convertToDate_Simple(stocks.get(i).getDate());

            // Check if the current date matches the last date
            if (current_date.equals(last_date)) {
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
                frame.add(stocks.get(i - 20 + j)); // Add stocks from i-20 to i-1
            }

            // Get notifications for the current frame
            List<Notification> notifications = getNotificationForFrame(frame);

            // Add notifications to alertsList if not empty
            if (!notifications.isEmpty()) {
                alertsList.addAll(notifications); // Add all notifications to alertsList
            }
        }

        return alertsList;
    }

    //!!!Finish percentage algorithm
    public static List<Notification> getNotificationForFrame(List<StockUnit> stocks) {
        List<Notification> alertsList = new ArrayList<>();

        double spikeThreshold = 0.3;      // Threshold for a spike in the short term
        double sustainedThreshold = 0.1;  // Threshold for a medium-term increase
        double volatilityThreshold = 0.05; // Minimum volatility level to detect a spike
        double cumulativeChangeThreshold = 1.0; // Minimum cumulative percentage change to confirm a spike

        double change = 0;
        double change0to7 = 0;
        double change7to14 = 0;
        double change14to20 = 0;
        double change17to20 = 0;

        List<Double> percentageChanges = new ArrayList<>(); // Store individual percentage changes for volatility calculation
        TimeSeries timeSeries = new TimeSeries("Nvidia stock");

        for (int i = 1; i < stocks.size(); i++) {
            double currentClose = stocks.get(i).getClose();
            double previousClose = stocks.get(i - 1).getClose();
            double percentageChange = ((currentClose - previousClose) / previousClose) * 100;
            timeSeries.add(new Minute(Main_data_handler.convertToDate(stocks.get(i).getDate())), stocks.get(i).getClose());

            percentageChanges.add(percentageChange); // Add to changes list for volatility calculation

            // Track cumulative changes across time frames
            if (i < 7) {
                change0to7 += percentageChange;
            } else if (i >= 7 && i < 14) {
                change7to14 += percentageChange;
            } else if (i >= 14 && i < 20) {
                change14to20 += percentageChange;
            }
            if (i >= 17) {
                change17to20 += percentageChange;
            }
            change += percentageChange;
        }

        // Calculate volatility as the standard deviation of percentage changes
        double volatility = calculateVolatility(percentageChanges);

        // Apply refined spike detection logic with new filters
        if (isRefinedSpikeEvent(change0to7, change7to14, change14to20, change17to20, change, spikeThreshold, sustainedThreshold, volatility, volatilityThreshold, cumulativeChangeThreshold)) {
            System.out.printf("%s, %.3f    %.3f    %.3f    %.3f    Volatility: %.3f%n", stocks.get(stocks.size() - 1).getDate(), change, change0to7, change7to14, change14to20, volatility);
            alertsList.add(new Notification("Nvidia stock" + stocks.get(stocks.size() - 1).getDate(), String.format("%s, %.3f    %.3f    %.3f    %.3f    Volatility: %.3f", stocks.get(stocks.size() - 1).getDate(), change, change0to7, change7to14, change14to20, volatility), timeSeries));
        }

        return alertsList;
    }

    // Helper method for calculating volatility (standard deviation of percentage changes)
    private static double calculateVolatility(List<Double> changes) {
        double mean = changes.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = changes.stream().mapToDouble(change -> Math.pow(change - mean, 2)).average().orElse(0.0);
        return Math.sqrt(variance); // Standard deviation as volatility measure
    }

    // Enhanced spike detection with additional filters
    private static boolean isRefinedSpikeEvent(double change0to7, double change7to14, double change14to20, double change17to20,
                                               double cumulativeChange, double spikeThreshold, double sustainedThreshold,
                                               double volatility, double volatilityThreshold, double cumulativeChangeThreshold) {
        int positivePeriodCount = 0;

        // Count periods with sustained positive changes
        if (change0to7 > sustainedThreshold) positivePeriodCount++;
        if (change7to14 > sustainedThreshold) positivePeriodCount++;
        if (change14to20 > sustainedThreshold) positivePeriodCount++;

        // Spike detection based on refined criteria:
        return (positivePeriodCount >= 2)                      // At least two periods show sustained positive change
                && (change17to20 >= spikeThreshold || change14to20 >= spikeThreshold)  // Significant recent spike
                && (volatility >= volatilityThreshold)          // Ensure sufficient volatility
                && (cumulativeChange >= cumulativeChangeThreshold) // Cumulative change meets threshold
                && (change14to20 > -0.1);                       // Prevents detection during significant downtrends
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

    public static void Stock_change(List<StockUnit> stocks) { //plot the stock percentage change
        // Create a TimeSeries object for plotting
        TimeSeries timeSeries = new TimeSeries("NVDA Stock Price");

        for (int i = 1; i < stocks.size(); i++) {
            String date = stocks.get(i).getDate();

            double currentClose = stocks.get(i).getClose();  // Get the current close price
            double previousClose = stocks.get(i - 1).getClose();  // Get the previous close price

            // Calculate percentage change between consecutive time points
            double percentageChange = ((currentClose - previousClose) / previousClose) * 100;

            if (percentageChange < 1.5 && percentageChange > -1.5) {
                timeSeries.add(new Minute(Main_data_handler.convertToDate(date)), percentageChange);
            }
        }

        // Plot the data
        Main_data_handler.plotData(timeSeries, "NVDA percentage change", "Date", "Percentage change");
    }

    public static List<Notification> Main_data_puller() throws IOException { //get stock notifications
        List<StockUnit> stocks = readStockUnitsFromFile("NVDA.txt"); //Get the Stock data from the file (simulate real Stock data)
        return tester(stocks); //test method to test the Stock data
    }
}

//TODO
//!!!Finish percentage algorithm