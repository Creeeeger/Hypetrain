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

    public static List<Notification> Main_data_puller() throws IOException {
        List<StockUnit> stocks = readStockUnitsFromFile("NVDA.txt"); //Get the Stock data from the file (simulate real Stock data)
        return tester(stocks); //test method to test the Stock data
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
        inter_day_stocks = get_Inter_Day(stocks, Main_data_handler.convertToDate_Simple(stocks.get(17000).getDate()));

        alerts = get_alerts_from_stock(inter_day_stocks);

        Stock_value(inter_day_stocks);
        Stock_change(inter_day_stocks);

        return alerts;
    }

    public static List<Notification> get_alerts_from_stock(List<StockUnit> stocks) {
        List<Notification> alertsList = new ArrayList<>();

        // Thresholds for detecting potential spikes or dips
        double consistencyThreshold = 0.2;  // Minimum percentage change per minute
        double toleranceThreshold = 0.05;   // Tolerance for minor fluctuations
        int consecutiveCount = 1;           // Number of consecutive minute changes required to trigger an alert

        int upCount = 0;    // Counter for consecutive upward movements
        int downCount = 0;  // Counter for consecutive downward movements

        // Loop through the Stock (minute-level data assumed)
        for (int i = 1; i < stocks.size(); i++) {
            double currentClose = stocks.get(i).getClose();  // Get the current close price
            double previousClose = stocks.get(i - 1).getClose();  // Get the previous close price

            // Calculate percentage change between consecutive time points
            double percentageChange = ((currentClose - previousClose) / previousClose) * 100;

            // Detect consistent upward movement (potential spike)
            if (percentageChange >= consistencyThreshold) {
                upCount++;  // Increment upward counter
                downCount = 0;  // Reset downward counter
            }
            // Detect consistent downward movement (potential dip)
            else if (percentageChange <= -consistencyThreshold) {
                downCount++;  // Increment downward counter
                upCount = 0;  // Reset upward counter
            }
            // Ignore minor fluctuations (inside the tolerance range)
            else if (Math.abs(percentageChange) < toleranceThreshold) {
            }
            // Reset counts if the change is significant but doesn't meet spike/dip criteria
            else {
                upCount = 0;
                downCount = 0;
            }

            //!!!fix the graph logic later for live view
            // If we have enough consecutive upward movements, trigger a spike notification
            if (upCount >= consecutiveCount) {
                TimeSeries timeSeries = new TimeSeries("View window of change");

                try {
                    // Check if we have enough previous data points to generate the view window
                    if (i >= 15) {
                        for (int j = -15; j < 5; j++) {
                            timeSeries.add(new Minute(Main_data_handler.convertToDate(stocks.get(i + j).getDate())), stocks.get(i + j).getClose());
                        }
                    } else {
                        // Handle the case where there's not enough data for a full window
                        for (int j = -i; j < 5; j++) {  // Start from the available data
                            timeSeries.add(new Minute(Main_data_handler.convertToDate(stocks.get(i + j).getDate())), stocks.get(i + j).getClose());
                        }
                    }

                    // Add the spike notification to the list
                    alertsList.add(Main_data_handler.create_Notification(true, Main_UI.selected_stock, percentageChange, timeSeries, currentClose, Main_data_handler.convertToDate(stocks.get(i).getDate())));

                } catch (Exception e) {
                    System.err.println("Error while generating spike notification: " + e.getMessage());
                    e.printStackTrace();  // Optionally log the full stack trace
                }

                upCount = 0;  // Reset the upward counter after the notification
            }

            // If we have enough consecutive downward movements, trigger a dip notification
            if (downCount >= consecutiveCount) {
                TimeSeries timeSeries = new TimeSeries("View window of change");

                try {
                    // Check if we have enough previous data points to generate the view window
                    if (i >= 15) {
                        for (int j = -15; j < 5; j++) {
                            timeSeries.add(new Minute(Main_data_handler.convertToDate(stocks.get(i + j).getDate())), stocks.get(i + j).getClose());
                        }

                    } else {
                        // Handle the case where there's not enough data for a full window
                        for (int j = -i; j < 5; j++) {  // Start from the available data
                            timeSeries.add(new Minute(Main_data_handler.convertToDate(stocks.get(i + j).getDate())), stocks.get(i + j).getClose());
                        }
                    }

                    alertsList.add(Main_data_handler.create_Notification(false, Main_UI.selected_stock, percentageChange, timeSeries, currentClose, Main_data_handler.convertToDate(stocks.get(i).getDate())));  // Add the spike notification to the list

                } catch (Exception e) {
                    System.err.println("Error while generating spike notification: " + e.getMessage());
                    e.printStackTrace();  // Optionally log the full stack trace
                }
                downCount = 0;  // Reset the downward counter after the notification
            }
        }

        // Return the list of notifications generated (spike/dip alerts)
        return alertsList;
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

    public static void Stock_value(List<StockUnit> stocks) {
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

    public static void Stock_change(List<StockUnit> stocks) {
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
}

//TODO
//!!!fix the graph logic later for live view