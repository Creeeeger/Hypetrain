package org.crecker;

import com.crazzyghost.alphavantage.AlphaVantage;
import com.crazzyghost.alphavantage.AlphaVantageException;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.fundamentaldata.response.CompanyOverview;
import com.crazzyghost.alphavantage.fundamentaldata.response.CompanyOverviewResponse;
import com.crazzyghost.alphavantage.news.response.NewsResponse;
import com.crazzyghost.alphavantage.parameters.Interval;
import com.crazzyghost.alphavantage.parameters.OutputSize;
import com.crazzyghost.alphavantage.stock.response.StockResponse;
import com.crazzyghost.alphavantage.timeseries.response.QuoteResponse;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import com.crazzyghost.alphavantage.timeseries.response.TimeSeriesResponse;
import org.jetbrains.annotations.NotNull;
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
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;

public class Main_data_handler {
    public static void main(String[] args) {
        // Replace with your actual Alpha Vantage API key since this is a free key
        String apiKey = "2NN1RGFV3V34ORCZ"; //SIKE NEW KEY

        // Configure the API client
        Config cfg = Config.builder()
                .key(apiKey)
                .timeOut(10) // Timeout in seconds
                .build();

        // Initialize the Alpha Vantage API
        AlphaVantage.api().init(cfg);

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

    public static void InitAPi(String token) {
        // Configure the API client
        Config cfg = Config.builder()
                .key(token)
                .timeOut(10) // Timeout in seconds
                .build();

        // Initialize the Alpha Vantage API
        AlphaVantage.api().init(cfg);
    }

    public static void get_timeline(String symbol_name, TimelineCallback callback) {
        List<StockUnit> stocks = new ArrayList<>(); // Directly use a List<StockUnit>

        AlphaVantage.api()
                .timeSeries()
                .intraday()
                .forSymbol(symbol_name)
                .interval(Interval.ONE_MIN)
                .outputSize(OutputSize.FULL)
                .onSuccess(e -> {
                    TimeSeriesResponse response = (TimeSeriesResponse) e;
                    stocks.addAll(response.getStockUnits()); // Populate the list

                    callback.onTimeLineFetched(stocks); // Call the callback with the Stock list
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void get_Info_Array(String symbol_name, DataCallback callback) {
        Double[] data = new Double[9];

        // Fetch fundamental data
        AlphaVantage.api()
                .fundamentalData()
                .companyOverview()
                .forSymbol(symbol_name)
                .onSuccess(e -> {
                    CompanyOverviewResponse overview_response = (CompanyOverviewResponse) e;
                    CompanyOverview response = overview_response.getOverview();

                    if (response != null) {
                        data[4] = response.getPERatio();
                        data[5] = response.getPEGRatio();
                        data[6] = response.getFiftyTwoWeekHigh();
                        data[7] = response.getFiftyTwoWeekLow();
                        data[8] = Double.valueOf(response.getMarketCapitalization());
                    } else {
                        System.out.println("Company overview response is null.");
                    }
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();

        AlphaVantage.api()
                .timeSeries()
                .quote()
                .forSymbol(symbol_name)
                .onSuccess(e -> {
                    QuoteResponse response = (QuoteResponse) e;

                    if (response != null) {
                        data[0] = response.getOpen();
                        data[1] = response.getHigh();
                        data[2] = response.getLow();
                        data[3] = response.getVolume();
                    } else {
                        System.out.println("Quote response is null.");
                    }

                    // Call the callback with the fetched data
                    callback.onDataFetched(data);
                })
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void handleSuccess(TimeSeriesResponse response) throws IOException {
        // This generates some test data since we don't have unlimited API access
        BufferedWriter bufferedWriter = getBufferedWriter(response); //in reversed format (new to old)
        bufferedWriter.close(); // Close the BufferedWriter to free system resources

        // Create a TimeSeries object for plotting
        TimeSeries timeSeries = new TimeSeries(response.getMetaData().getSymbol().toUpperCase() + " Stock Price");

        // Get Stock units
        List<StockUnit> stocks = response.getStockUnits();

        Collections.reverse(stocks); //reverse (old to new)

        // Populate the time series with Stock data
        for (StockUnit stock : stocks) {
            String timestamp = stock.getDate();
            double closingPrice = stock.getClose(); // Assuming getClose() returns closing price

            // Add the data to the TimeSeries
            timeSeries.add(new Minute(convertToDate(timestamp)), closingPrice);
        }

        // Plot the data
        plotData(timeSeries, response.getMetaData().getSymbol().toUpperCase() + " Stock prices", "Date", "Price");
    }

    private static BufferedWriter getBufferedWriter(TimeSeriesResponse response) throws IOException {
        File data = new File(response.getMetaData().getSymbol().toUpperCase() + ".txt"); // Create a File object for the output file named "NVDA.txt"

        // Check if the file already exists
        if (!data.exists()) {
            // If the file does not exist, create a new file
            data.createNewFile(); // May throw IOException if it fails
        }

        // Initialize BufferedWriter to write to the file
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(data)); // Create a BufferedWriter to write to the file
        bufferedWriter.write(Arrays.toString(response.getStockUnits().toArray())); // Write the Stock units data to the file as a string
        bufferedWriter.flush(); // Flush the writer to ensure all data is written to the file
        return bufferedWriter;
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
        chartPanel.setMouseWheelEnabled(true); // Zoom with mouse wheel
        chartPanel.setZoomAroundAnchor(true);  // Zoom on the point where the mouse is anchored
        chartPanel.setRangeZoomable(true);     // Allow zooming on the Y-axis
        chartPanel.setDomainZoomable(true);    // Allow zooming on the X-axis

        // Add buttons for controlling the scale
        JPanel controlPanel = getjPanel(chartPanel);

        // Create the frame to display the chart
        JFrame frame = new JFrame("Stock Data");
        frame.setLayout(new BorderLayout());
        frame.add(chartPanel, BorderLayout.CENTER);
        frame.add(controlPanel, BorderLayout.SOUTH); // Add control buttons to the frame
        frame.pack();
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    @NotNull
    private static JPanel getjPanel(ChartPanel chartPanel) {
        JButton autoRangeButton = new JButton("Auto Range");
        autoRangeButton.addActionListener(e -> chartPanel.restoreAutoBounds()); // Reset to original scale

        JButton zoomInButton = new JButton("Zoom In");
        zoomInButton.addActionListener(e -> chartPanel.zoomInBoth(0.5, 0.5)); // Zoom in by 50%

        JButton zoomOutButton = new JButton("Zoom Out");
        zoomOutButton.addActionListener(e -> chartPanel.zoomOutBoth(0.5, 0.5)); // Zoom out by 50%

        JPanel controlPanel = new JPanel();

        controlPanel.add(autoRangeButton);
        controlPanel.add(zoomInButton);
        controlPanel.add(zoomOutButton);
        return controlPanel;
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
        // Assuming timestamp format is "yyyy-MM-dd HH:mm:ss" to "yyyy-MM-dd" date
        try {
            return new SimpleDateFormat("yyyy-MM-dd").parse(timestamp);
        } catch (ParseException e) {
            e.printStackTrace();
            return new Date();
        }
    }

    public static Notification create_Notification(boolean spike, String company, double percentage, TimeSeries timeSeries, double close, Date date) {
        String title, content;
        if (spike) {
            title = String.format("%.2f%% spike for %s", percentage, company);
            content = String.format("Consistent upward movement of %.2f%%, Price: %.2f, Date: %s", percentage, close, date);

        } else {
            title = String.format("%.2f%% dip for %s", percentage, company);
            content = String.format("Consistent downward movement of %.2f%%, Price: %.2f, Date: %s", percentage, close, date);
        }

        return new Notification(title, content, timeSeries);
    }

    public static void handleFailure(AlphaVantageException error) {
        System.out.println("error: " + error.getMessage());
    }

    public static void findMatchingSymbols(String searchText, SymbolSearchCallback callback) {
        List<String> allSymbols = new ArrayList<>();

        AlphaVantage.api()
                .Stocks()
                .setKeywords(searchText)
                .onSuccess(e -> {
                    List<StockResponse.StockMatch> list = e.getMatches();

                    for (StockResponse.StockMatch stockResponse : list) {
                        allSymbols.add(stockResponse.getSymbol());
                    }
                    // Filter and invoke the callback on success
                    List<String> filteredSymbols = allSymbols.stream()
                            .filter(symbol -> symbol.toUpperCase().startsWith(searchText.toUpperCase()))
                            .collect(Collectors.toList());
                    callback.onSuccess(filteredSymbols);
                })
                .onFailure(failure -> {
                    // Handle failure and invoke the failure callback
                    Main_data_handler.handleFailure(failure);
                    callback.onFailure(new RuntimeException("API call failed"));
                })
                .fetch();
    }

    public static void receive_News(String Symbol, ReceiveNewsCallback callback) {
        AlphaVantage.api()
                .News()
                .setTickers(Symbol)
                .setSort("LATEST")
                .setLimit(12)
                .onSuccess(e -> callback.onNewsReceived(e.getNewsItems()))
                .onFailure(Main_data_handler::handleFailure)
                .fetch();
    }

    public static void start_Hype_Mode(int Volume, float Hype) {
        System.out.printf("Settings: %s Volume, %s Hype%n", Volume, Hype);
        int price_per_stock = 1; //!!!Update to real price
        int amt_to_buy = Volume / price_per_stock;
        //!!!Add logic for hype mode
    }

    //!!!finish get_available_symbols method
    public static List<String> get_available_symbols(int volume) { // Method for receiving trade-able symbols for amount of volume
        List<String> symbols = new ArrayList<>(); // Initialize an ArrayList to hold the symbols

//        AlphaVantage.api()
//                .stocks()
//                .forVolume(volume)
//                .onSuccess(e -> {
//                    List<StockResponse.StockMatch> results = e.getMatches();
//                    for (StockResponse.StockMatch match : results) {
//                        symbols.add(match.getSymbol()); // Add each symbol to the ArrayList
//                    }
//                })
//                .onFailure(Main_data_handler::handleFailure)
//                .fetch();

        return symbols; // Return the ArrayList of symbols
    }

    //Interfaces
    public interface DataCallback {
        void onDataFetched(Double[] values);
    }

    public interface TimelineCallback {
        void onTimeLineFetched(List<StockUnit> stocks);
    }

    public interface SymbolSearchCallback {
        void onSuccess(List<String> symbols);

        void onFailure(Exception e);
    }

    public interface ReceiveNewsCallback {
        void onNewsReceived(List<NewsResponse.NewsItem> news);
    }
}

//TODO
//!!!Add logic for hype mode
//!!!finish get_available_symbols method for volume filtering