package org.crecker;

import com.crazzyghost.alphavantage.AlphaVantage;
import com.crazzyghost.alphavantage.Config;
import com.crazzyghost.alphavantage.parameters.Interval;
import com.crazzyghost.alphavantage.parameters.OutputSize;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import com.crazzyghost.alphavantage.timeseries.response.TimeSeriesResponse;
import org.jetbrains.annotations.NotNull;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.time.Minute;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.crecker.Main_data_handler.*;

public class data_tester {
    public static JLabel percentageChange;
    static JFreeChart chart;
    private static double point1X = Double.NaN;
    private static double point1Y = Double.NaN;
    private static double point2X = Double.NaN;
    private static double point2Y = Double.NaN;
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;

    //new data filler method
    public static void main(String[] args) {
        tester();
        // getData("WOLF");
    }

    //method for pulling new data from server for tests and training
    public static void getData(String symbol) {
        String apiKey = "2NN1RGFV3V34ORCZ";

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
                .forSymbol(symbol)
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
        BufferedWriter bufferedWriter = getBufferedWriter(response); //in reversed format (new to old)
        bufferedWriter.close(); // Close the BufferedWriter to free system resources
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

    public static void tester() {
        String[] fileNames = {"NVDA.txt", "PLTR.txt", "SMCI.txt", "TSLA.txt", "TSM.txt", "WOLF.txt", "MSTR.txt", "SNOW.txt"}; //add more files
        int stock = 2;

        // Process data for each file
        for (String fileName : fileNames) {
            try {
                processStockDataFromFile(fileName, fileName.substring(0, fileName.indexOf(".")));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        calculateStockPercentageChange();
        calculateSpikes();

        TimeSeries timeSeries = new TimeSeries("stock");
        for (int i = 2; i < stockList.size(); i++) {
            try {
                String timestamp = stockList.get(i).stockUnits.get(stock).getDate();
                double closingPrice = stockList.get(i).stockUnits.get(stock).getClose(); // Assuming getClose() returns closing price
                double prevClosingPrice = stockList.get(i - 1).stockUnits.get(stock).getClose(); // Assuming getClose() returns closing price

                if (Math.abs((closingPrice - prevClosingPrice) / prevClosingPrice * 100) > 14) {
                    closingPrice = prevClosingPrice;
                }

                // Add the data to the TimeSeries
                timeSeries.addOrUpdate(new Minute(convertToDate(timestamp)), closingPrice);
            } catch (Exception e) {
                break;
            }
        }

        plotData(timeSeries, stockList.get(4).stockUnits.get(stock).getSymbol() + " price change", "Date", "price");
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
        stockUnits.subList(0, stockUnits.size() - 6000).clear();

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

    public static void processStockDataFromFile(String filePath, String symbol) throws IOException {
        // Read stock data from file into stockUnits
        List<StockUnit> stockUnits = readStockUnitsFromFile(filePath);
        Main_data_handler.stock stockObj;

        // Ensure stockList is initialized
        if (stockList == null) {
            stockList = new ArrayList<>();
        }
        boolean isFirstRound = stockList.isEmpty();

        for (int i = 0; i < stockUnits.size(); i++) {
            List<StockUnit> stockBatch = new ArrayList<>();
            stockUnits.get(i).setSymbol(symbol);
            stockBatch.add(stockUnits.get(i));
            stockObj = new Main_data_handler.stock(new ArrayList<>(stockBatch));

            if (isFirstRound) {
                stockList.add(stockObj); // Add the stock object to stockList
            } else {
                if (i > 0 && i < stockList.size()) {  // Ensure you're not accessing invalid indices
                    int prevSize = stockList.get(i - 1).stockUnits.size();
                    int currSize = stockList.get(i).stockUnits.size();

                    if (prevSize != currSize + 1) {
                        // Handle the mismatch, possibly adding the stock unit to the current stock object
                        if (currSize > 0) {
                            stockList.get(i).stockUnits.add(stockUnits.get(i));
                        }

                    } else {
                        // Normal case where no mismatch
                        stockList.get(i).stockUnits.add(stockUnits.get(i));
                    }
                }
            }
        }
    }

    public static void calculateStockPercentageChange() {
        // Check if there are at least two batches of stock data in stockList
        for (int i = 1; i < stockList.size(); i++) {
            // Get the last two batches
            Main_data_handler.stock currentStockBatch = stockList.get(i);
            Main_data_handler.stock previousStockBatch = stockList.get(i - 1);

            // Check if the current batch and previous batch have the same number of stock units
            if (currentStockBatch.stockUnits.size() == previousStockBatch.stockUnits.size()) {
                // Iterate through the stock units in the current and previous batches
                for (int j = 0; j < currentStockBatch.stockUnits.size(); j++) {
                    // Get the current and previous stock units
                    StockUnit currentStockUnit = currentStockBatch.stockUnits.get(j);
                    StockUnit previousStockUnit = previousStockBatch.stockUnits.get(j);

                    // Get the current close price and the previous close price
                    double currentClose = currentStockUnit.getClose();
                    double previousClose = previousStockUnit.getClose();

                    // Calculate the percentage change between the consecutive stock units
                    double percentageChange = ((currentClose - previousClose) / previousClose) * 100;

                    // Check for a 10% dip or peak
                    if (Math.abs(percentageChange) >= 14) {
                        currentStockUnit.setPercentageChange(previousStockUnit.getPercentageChange());

                    } else {
                        // Set the percentage change using the setter method
                        currentStockUnit.setPercentageChange(percentageChange);
                    }
                }
            }
        }
    }

    public static void plotData(TimeSeries timeSeries, String chart_name, String X_axis, String Y_axis) {
        // Wrap the TimeSeries in a TimeSeriesCollection, which implements XYDataset
        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(timeSeries);

        // Create the chart with the dataset
        chart = ChartFactory.createTimeSeriesChart(
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

        chartPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Point2D p = chartPanel.translateScreenToJava2D(e.getPoint());
                XYPlot plot = chart.getXYPlot();
                ValueAxis xAxis = plot.getDomainAxis();
                ValueAxis yAxis = plot.getRangeAxis();

                // Convert mouse position to data coordinates
                double x = xAxis.java2DToValue(p.getX(), chartPanel.getScreenDataArea(), plot.getDomainAxisEdge());
                double y = yAxis.java2DToValue(p.getY(), chartPanel.getScreenDataArea(), plot.getRangeAxisEdge());

                if (Double.isNaN(point1X)) {
                    // First point selected, set the first marker
                    point1X = x;
                    point1Y = y;
                    addFirstMarker(plot, point1X);
                } else {
                    // Second point selected, set the second marker and shaded region
                    point2X = x;
                    point2Y = y;
                    addSecondMarkerAndShade(plot);

                    // Reset points for next selection
                    point1X = Double.NaN;
                    point1Y = Double.NaN;
                    point2X = Double.NaN;
                    point2Y = Double.NaN;
                }
            }
        });

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

    private static void addFirstMarker(XYPlot plot, double xPosition) {
        // Clear previous markers if any
        if (marker1 != null) {
            plot.removeDomainMarker(marker1);
        }
        if (marker2 != null) {
            plot.removeDomainMarker(marker2);
        }
        if (shadedRegion != null) {
            plot.removeDomainMarker(shadedRegion);
        }

        // Create and add the first marker
        marker1 = new ValueMarker(xPosition);
        marker1.setPaint(Color.GREEN);  // First marker in green
        marker1.setStroke(new BasicStroke(1.5f));  // Customize thickness
        plot.addDomainMarker(marker1);
    }

    private static void addSecondMarkerAndShade(XYPlot plot) {
        // Clear previous markers and shaded region if they exist
        if (marker2 != null) {
            plot.removeDomainMarker(marker2);
        }
        if (shadedRegion != null) {
            plot.removeDomainMarker(shadedRegion);
        }

        // Calculate the percentage difference between the two y-values
        double percentageDiff = ((point2Y - point1Y) / point1Y) * 100;

        // Determine the color of the second marker based on percentage difference
        Color markerColor = (percentageDiff >= 0) ? Color.GREEN : Color.RED;
        Color shadeColor = (percentageDiff >= 0) ? new Color(100, 200, 100, 50) : new Color(200, 100, 100, 50);

        // Create and add the second marker
        marker2 = new ValueMarker(point2X);
        marker2.setPaint(markerColor);
        marker2.setStroke(new BasicStroke(1.5f)); // Customize thickness
        plot.addDomainMarker(marker2);

        // Create and add the shaded region between the two markers
        shadedRegion = new IntervalMarker(Math.min(point1X, point2X), Math.max(point1X, point2X));
        shadedRegion.setPaint(shadeColor);  // Translucent green or red shade
        plot.addDomainMarker(shadedRegion);
        percentageChange.setText(String.format("Percentage Change: %.3f%%", percentageDiff));
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
        percentageChange = new JLabel("Percentage Change");

        controlPanel.add(autoRangeButton);
        controlPanel.add(zoomInButton);
        controlPanel.add(zoomOutButton);
        controlPanel.add(percentageChange);
        return controlPanel;
    }
}