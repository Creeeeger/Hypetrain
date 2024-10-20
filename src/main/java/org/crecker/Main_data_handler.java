package org.crecker;

import com.crazzyghost.alphavantage.AlphaVantage;
import com.crazzyghost.alphavantage.AlphaVantageException;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.parameters.Interval;
import com.crazzyghost.alphavantage.parameters.OutputSize;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import com.crazzyghost.alphavantage.timeseries.response.TimeSeriesResponse;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

public class Main_data_handler {
    public static void main(String[] args) {
        // Replace with your actual Alpha Vantage API key
        String apiKey = "0988PSIKXZ50IP2T";

        // Configure the API client
        Config cfg = Config.builder()
                .key(apiKey)
                .timeOut(10) // Timeout in seconds
                .build();

        // Initialize the Alpha Vantage API
        AlphaVantage.api().init(cfg);

        AlphaVantage.api().timeSeries().intraday().interval(Interval.ONE_MIN).fetch();

        AlphaVantage.api()
                .timeSeries()
                .intraday()
                .forSymbol("NVDA")
                .interval(Interval.ONE_MIN)
                .outputSize(OutputSize.FULL)
                .onSuccess(e -> {
                    try {
                        handleSuccess((TimeSeriesResponse) e);
                    } catch (IOException ex) {
                        throw new RuntimeException(ex);
                    }
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void handleSuccess(TimeSeriesResponse response) throws IOException {
        // Print out the stock units for debugging
        System.out.println(Arrays.toString(response.getStockUnits().toArray()));

        // This generates some test data since we don't have unlimited API access
        File data = new File("NVDA.txt"); // Create a File object for the output file named "NVDA.txt"

        // Check if the file already exists
        if (!data.exists()) {
            // If the file does not exist, create a new file
            data.createNewFile(); // May throw IOException if it fails
        }

        // Initialize BufferedWriter to write to the file
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(data)); // Create a BufferedWriter to write to the file
        bufferedWriter.write(Arrays.toString(response.getStockUnits().toArray())); // Write the stock units data to the file as a string
        bufferedWriter.flush(); // Flush the writer to ensure all data is written to the file
        bufferedWriter.close(); // Close the BufferedWriter to free system resources

        // Create a TimeSeries object for plotting
        TimeSeries timeSeries = new TimeSeries("NVDA Stock Price");

        // Get stock units
        List<StockUnit> stocks = response.getStockUnits();

        // Populate the time series with stock data
        for (StockUnit stock : stocks) {
            String timestamp = stock.getDate();
            double closingPrice = stock.getClose(); // Assuming getClose() returns closing price

            // Add the data to the TimeSeries
            timeSeries.add(new Minute(convertToDate(timestamp)), closingPrice);
        }

        // Plot the data
        plotData(timeSeries);
    }

    private static void plotData(TimeSeries timeSeries) {
        // Wrap the TimeSeries in a TimeSeriesCollection, which implements XYDataset
        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(timeSeries);

        // Create the chart with the dataset
        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                "NVDA Stock Prices", // Chart title
                "Time", // X-axis Label
                "Price", // Y-axis Label
                dataset, // The dataset (now a TimeSeriesCollection)
                true, // Show legend
                true, // Show tooltips
                false // Show URLs
        );

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(800, 600));

        JFrame frame = new JFrame("Stock Data");
        frame.setContentPane(chartPanel);
        frame.pack();
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    private static Date convertToDate(String timestamp) {
        // Convert timestamp to Date
        // Assuming timestamp format is "yyyy-MM-dd HH:mm:ss"
        try {
            return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(timestamp);
        } catch (ParseException e) {
            e.printStackTrace();
            return new Date(); // Return current date if parsing fails
        }
    }

    public static void handleFailure(AlphaVantageException error) {
        System.out.println("error" + error.getMessage());
    }
}