package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class data_tester {
    static List<StockUnit> inter_day_stocks;
    static List<Notification> alerts;

    public static void main(String[] args) throws IOException {
        List<StockUnit> stocks = readStockUnitsFromFile("NVDA.txt"); //Get the stock data from the file (simulate real stock data)
        tester(stocks); //test method to test the stock data

        //further code to test on data comes here
        System.out.println("Data got loaded successfully!");
    }

    public static List<Notification> Main_data_puller() throws IOException {
        List<StockUnit> stocks = readStockUnitsFromFile("NVDA.txt"); //Get the stock data from the file (simulate real stock data)
        return tester(stocks); //test method to test the stock data
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

        return stockUnits;
    }

    private static StockUnit parseStockUnit(String stockUnitString) {
        // Remove "StockUnit{" from the beginning of the string
        stockUnitString = stockUnitString.replace("StockUnit{", "").trim();

        // Split the stock unit attributes by commas
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
        inter_day_stocks = get_Inter_Day(stocks, Main_data_handler.convertToDate_Simple(stocks.get(3000).getDate()));

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

        // Loop through the stocks (minute-level data assumed)
        for (int i = 1; i < stocks.size(); i++) {
            String date = stocks.get(i).getDate();  // Get the date of the current stock entry
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

            // If we have enough consecutive upward movements, trigger a spike notification
            if (upCount >= consecutiveCount) {
                String title = String.format("%.2f%% Spike!", percentageChange);
                String content = String.format("Consistent upward movement of %.2f%% over %d minutes as of %s. Closing price: %.2f",
                        percentageChange, consecutiveCount, date, currentClose);
                Notification alert = new Notification(title, content);
                alertsList.add(alert);  // Add the spike notification to the list
                upCount = 0;  // Reset the upward counter after the notification
            }

            // If we have enough consecutive downward movements, trigger a dip notification
            if (downCount >= consecutiveCount) {
                String title = String.format("%.2f%% Dip!", percentageChange);
                String content = String.format("Consistent downward movement of %.2f%% over %d minutes as of %s. Closing price: %.2f",
                        percentageChange, consecutiveCount, date, currentClose);
                Notification alert = new Notification(title, content);
                alertsList.add(alert);  // Add the dip notification to the list
                downCount = 0;  // Reset the downward counter after the notification
            }
        }

        // Return the list of notifications generated (spike/dip alerts)
        return alertsList;
    }

    public static List<StockUnit> get_Inter_Day(List<StockUnit> stocks, Date last_date) {
        List<StockUnit> inter_day_stocks = new ArrayList<>();

        for (int i = 1; i < stocks.size(); i++) {
            Date current_date = Main_data_handler.convertToDate_Simple(stocks.get(i).getDate());

            if (current_date.equals(last_date)) {
                double current_close = stocks.get(i).getClose();
                double previous_close = stocks.get(i - 1).getClose();

                // Check for a 10% dip or peak
                if (Math.abs((current_close - previous_close) / previous_close) >= 0.1) {
                    // Replace the current close with the previous close
                    current_close = previous_close;
                }

                // Create a new StockUnit with the modified close value
                StockUnit newStock = new StockUnit.Builder()
                        .open(stocks.get(i).getOpen())
                        .high(stocks.get(i).getHigh())
                        .low(stocks.get(i).getLow())
                        .close(current_close)  // Use the updated close value
                        .adjustedClose(stocks.get(i).getAdjustedClose())
                        .volume(stocks.get(i).getVolume())
                        .dividendAmount(stocks.get(i).getDividendAmount())
                        .splitCoefficient(stocks.get(i).getSplitCoefficient())
                        .time(stocks.get(i).getDate())
                        .build();

                // Replace the old stock entry with the new one
                stocks.set(i, newStock);

                // Add the new stock to the inter_day_stocks list
                inter_day_stocks.add(newStock);
            }
        }
        return inter_day_stocks;
    }

    public static void Stock_value(List<StockUnit> stocks) {
        // Create a TimeSeries object for plotting
        TimeSeries timeSeries = new TimeSeries("NVDA Stock Price");

        // Populate the time series with stock data
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

        for (int i = 1; i < stocks.size() - 1; i++) {
            String date = stocks.get(i).getDate();

            double first_val = stocks.get(i).getClose();
            double second_val = stocks.get(i - 1).getClose();

            double percentage_change = ((second_val / first_val) * 100) - 100;

            if (percentage_change < 1.5 && percentage_change > -1.5) {
                timeSeries.add(new Minute(Main_data_handler.convertToDate(date)), percentage_change);
            }
        }

        // Plot the data
        Main_data_handler.plotData(timeSeries, "NVDA percentage change", "Date", "Percentage change");
    }
}
