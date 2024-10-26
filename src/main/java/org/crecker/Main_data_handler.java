package org.crecker;

import com.crazzyghost.alphavantage.AlphaVantage;
import com.crazzyghost.alphavantage.AlphaVantageException;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.parameters.Interval;
import com.crazzyghost.alphavantage.parameters.OutputSize;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import com.crazzyghost.alphavantage.timeseries.response.TimeSeriesResponse;
import com.toedter.calendar.JDateChooser;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
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
import java.util.stream.Collectors;

public class Main_data_handler {
    public static JDateChooser dateChooser_to, dateChooser_from;

    public static void main(String[] args) {
        // Replace with your actual Alpha Vantage API key since this is a free key
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
        plotData(timeSeries, "NVDA Stock prices", "Date", "Price");
    }

    public static void plotData(TimeSeries timeSeries, String chart_name, String X_axis, String Y_axis) {
        // Wrap the TimeSeries in a TimeSeriesCollection, which implements XYDataset
        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(timeSeries);

        // Create the chart with the dataset
        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                chart_name, // Chart title
                X_axis, // X-axis Label
                Y_axis, // Y-axis Label
                dataset, // The dataset (now a TimeSeriesCollection)
                true, // Show legend
                true, // Show tooltips
                false // Show URLs
        );

        // Customizing the plot
        XYPlot plot = chart.getXYPlot();

        // Add a light red shade below Y=0
        plot.addRangeMarker(new IntervalMarker(Double.NEGATIVE_INFINITY, 0.0, new Color(255, 200, 200, 100)));

        // Add a light green shade above Y=0
        plot.addRangeMarker(new IntervalMarker(0.0, Double.POSITIVE_INFINITY, new Color(200, 255, 200, 100)));

        // Add a black line at y = 0
        ValueMarker zeroLine = new ValueMarker(0.0);
        zeroLine.setPaint(Color.BLACK); // Set the color to black
        zeroLine.setStroke(new BasicStroke(1.0f)); // Set the stroke for the line
        plot.addRangeMarker(zeroLine);

        // Enable zoom and pan features on the chart panel
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(1600, 600));
        chartPanel.setMouseWheelEnabled(true); // Zoom with mouse wheel
        chartPanel.setZoomAroundAnchor(true);  // Zoom on the point where the mouse is anchored
        chartPanel.setRangeZoomable(true);     // Allow zooming on the Y-axis
        chartPanel.setDomainZoomable(true);    // Allow zooming on the X-axis

        // Add buttons for controlling the scale
        JButton autoRangeButton = new JButton("Auto Range");
        autoRangeButton.addActionListener(e -> chartPanel.restoreAutoBounds()); // Reset to original scale

        JButton zoomInButton = new JButton("Zoom In");
        zoomInButton.addActionListener(e -> chartPanel.zoomInBoth(0.5, 0.5)); // Zoom in by 50%

        JButton zoomOutButton = new JButton("Zoom Out");
        zoomOutButton.addActionListener(e -> chartPanel.zoomOutBoth(0.5, 0.5)); // Zoom out by 50%

        JPanel controlPanel = new JPanel();
        JLabel from = new JLabel("Date from: ");
        dateChooser_from = new JDateChooser();  // Create the date chooser

        JLabel to = new JLabel("Date to: ");
        dateChooser_to = new JDateChooser();

        controlPanel.add(from);
        controlPanel.add(dateChooser_from);

        controlPanel.add(to);
        controlPanel.add(dateChooser_to);

        controlPanel.add(autoRangeButton);
        controlPanel.add(zoomInButton);
        controlPanel.add(zoomOutButton);

        dateChooser_from.addPropertyChangeListener(evt -> {
            Date date = dateChooser_from.getDate();
            if (date != null) {
                System.out.println(date);
            }
        });

        dateChooser_to.addPropertyChangeListener(evt -> {
            Date date = dateChooser_to.getDate();
            if (date != null) {
                System.out.println(date);
            }
        });

        // Create the frame to display the chart
        JFrame frame = new JFrame("Stock Data");
        frame.setLayout(new BorderLayout());
        frame.add(chartPanel, BorderLayout.CENTER);
        frame.add(controlPanel, BorderLayout.SOUTH); // Add control buttons to the frame
        frame.pack();
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static Date convertToDate(String timestamp) {
        // Convert timestamp to Date
        // Assuming timestamp format is "yyyy-MM-dd HH:mm:ss"
        try {
            return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(timestamp);
        } catch (ParseException e) {
            e.printStackTrace();
            return new Date();
        }
    }

    public static Date convertToDate_Simple(String timestamp) {
        // Convert timestamp to Date
        // Assuming timestamp format is "yyyy-MM-dd HH:mm:ss"
        try {
            return new SimpleDateFormat("yyyy-MM-dd").parse(timestamp);
        } catch (ParseException e) {
            e.printStackTrace();
            return new Date();
        }
    }

    public static void handleFailure(AlphaVantageException error) {
        System.out.println("error" + error.getMessage());
    }

    public static List<String> findMatchingSymbols(String searchText) {
        //!!!Add logic to add the real symbols from the api
        List<String> allSymbols = List.of("AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"); // Example stock symbols


        return allSymbols.stream()
                .filter(symbol -> symbol.toUpperCase().startsWith(searchText.toUpperCase()))
                .collect(Collectors.toList());
    }

    public static void start_Hype_Mode(int Volume, float Hype) {
        System.out.printf("Settings: %s Volume, %s Hype%n", Volume, Hype);
        //!!!Add logic for hype mode
    }
}

//TODO
//!!!Add logic to add the real symbols from the api
//!!!Add logic for hype mode