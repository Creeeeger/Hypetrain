package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jetbrains.annotations.NotNull;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.time.Minute;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static org.crecker.Main_UI.addNotification;
import static org.crecker.Main_UI.gui;
import static org.crecker.Main_data_handler.*;
import static org.crecker.data_tester.getData;
import static org.crecker.data_tester.parseStockUnit;

public class pLTester {
    // Index map for quick timestamp lookups
    private static final Map<String, Map<LocalDateTime, Integer>> symbolTimeIndex = new ConcurrentHashMap<>();
    public static boolean debug = true; // Flag for printing PL
    public static JLabel percentageChange;
    static boolean saveData = false;
    static double spikeThreshold = 0.8;
    static int lookBackWindow = 5;
    private static double point1X = Double.NaN;
    private static double point1Y = Double.NaN;
    private static double point2X = Double.NaN;
    private static double point2Y = Double.NaN;
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;

    public static void main(String[] args) {
        //updateStocks();
        long startTime = System.nanoTime(); // Record the start time

        PLAnalysis(); // Call the method you want to monitor

        long endTime = System.nanoTime(); // Record the end time
        long durationInNanoseconds = endTime - startTime;

        System.out.println("Execution time: " + (durationInNanoseconds / 1_000_000) + " milliseconds");
    }

    private static void updateStocks() {
        for (String stock : Arrays.asList("SMCI", "IONQ", "WOLF", "MARA")) {
            getData(stock);
        }
    }

    public static void PLAnalysis() {
        //   final String[] SYMBOLS = {"MARA.txt", "IONQ.txt", "SMCI.txt", "WOLF.txt"};
        final String[] SYMBOLS = {"MARA.txt"};

        final double INITIAL_CAPITAL = 130000;
        final int FEE = 0;
        final int stock = 0;
        final double DIP_LEVEL = -0.5;

        prepData(SYMBOLS, 2000);

        // Preprocess indices during data loading
        Arrays.stream(SYMBOLS).forEach(symbol -> buildTimeIndex(symbol.replace(".txt", ""),
                getSymbolTimeline(symbol.replace(".txt", ""))));

        double capital = INITIAL_CAPITAL;
        int successfulCalls = 0;
        LocalDateTime lastTradeTime = null;

        // Cache timelines per notification
        Map<String, List<StockUnit>> timelineCache = new HashMap<>();

        for (Notification notification : notificationsForPLAnalysis) {
            if (gui != null) {
                createNotification(notification);
            }

            String symbol = notification.getSymbol();
            List<StockUnit> timeline = timelineCache.computeIfAbsent(symbol, Main_data_handler::getSymbolTimeline);

            Integer index = getIndexForTime(symbol, notification.getLocalDateTime());

            if (index == null || index >= timeline.size() - 1) {
                System.out.println("Invalid time index for " + symbol);
                continue;
            }

            StockUnit nextUnit = timeline.get(index + 1);

            if (shouldProcessDip(nextUnit, DIP_LEVEL, lastTradeTime)) {
                TradeResult result = processTradeSequence(timeline, index + 2, DIP_LEVEL, capital, notification.getSymbol());
                capital = result.newCapital() - FEE;
                successfulCalls++;
                lastTradeTime = result.lastTradeTime();
                if (debug) {
                    logTradeResult(symbol, result);
                    getNext5Minutes(capital, lastTradeTime, notification.getSymbol());
                }
            }
        }
        if (debug) {
            createTimeline(SYMBOLS[stock]);
            logFinalResults(DIP_LEVEL, capital, INITIAL_CAPITAL, successfulCalls);
        }
    }

    private static void buildTimeIndex(String symbol, List<StockUnit> timeline) {
        Map<LocalDateTime, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < timeline.size(); i++) {
            indexMap.put(timeline.get(i).getLocalDateTimeDate(), i);
        }
        symbolTimeIndex.put(symbol, indexMap);
    }

    private static Integer getIndexForTime(String symbol, LocalDateTime time) {
        return symbolTimeIndex.getOrDefault(symbol, Collections.emptyMap()).get(time);
    }

    private static boolean shouldProcessDip(StockUnit nextUnit, double dipLevel, LocalDateTime lastTradeTime) {
        return nextUnit.getPercentageChange() >= dipLevel && (lastTradeTime == null || nextUnit.getLocalDateTimeDate().isAfter(lastTradeTime));
    }

    private static TradeResult processTradeSequence(List<StockUnit> timeline, int startIndex, double dipLevel, double capital, String symbol) {
        double currentCapital = capital;
        int currentIndex = startIndex;
        final int maxSteps = Math.min(timeline.size(), startIndex + 100); // Safety limit

        while (currentIndex < maxSteps) {
            StockUnit unit = timeline.get(currentIndex);
            currentCapital *= (1 + (unit.getPercentageChange() / 100));

            if (debug) {
                System.out.printf("%s trade: capital %.2f change %.2f Date: %s%n", symbol, capital, unit.getPercentageChange(), unit.getDateDate());
            }

            if (unit.getPercentageChange() < dipLevel) {
                break;
            }
            currentIndex++;
        }

        // catch errors in case not enough data points are available
        try {
            return new TradeResult(currentCapital, timeline.get(currentIndex).getLocalDateTimeDate());
        } catch (Exception e) {
            return new TradeResult(currentCapital, timeline.get(currentIndex - 1).getLocalDateTimeDate());
        }
    }

    private static void logTradeResult(String symbol, TradeResult result) {
        System.out.printf("%s trade: Final capital %.2f at %s%n",
                symbol,
                result.newCapital(),
                result.lastTradeTime().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
        );
    }

    private static void logFinalResults(double dipLevel, double capital, double initial, int calls) {
        double revenue = (capital - initial) * 0.75;
        if (calls > 0) {
            System.out.printf("Dip Level: %.2f%n", dipLevel);
            System.out.printf("Total Revenue: €%.2f%n", revenue);
            System.out.printf("Successful Calls: %d%n", calls);
            System.out.printf("Revenue/Call: €%.2f%n%n", revenue / calls);
        }
    }

    private static void prepData(String[] fileNames, int cut) {
        // Calculation of rallies, Process data for each file
        Arrays.stream(fileNames).forEach(fileName -> {
            try {
                processStockDataFromFile(fileName, fileName.substring(0, fileName.indexOf(".")), cut);
            } catch (IOException e) {
                e.printStackTrace();
            }
        });

        data_tester.calculateStockPercentageChange();
        rallyDetector(frameSize, false);
    }

    public static void processStockDataFromFile(String filePath, String symbol, int retainLast) throws IOException {
        List<StockUnit> fileUnits = readStockUnitsFromFile(filePath, retainLast);
        symbol = symbol.toUpperCase();

        // Engineer the target feature for the Data export for training
        fileUnits.forEach(unit -> unit.setTarget(0));

        // Iterate through all units to find spikes
        for (int i = 0; i < fileUnits.size() - lookBackWindow; i++) {
            int windowEnd = i + lookBackWindow - 1;
            StockUnit start = fileUnits.get(i);
            StockUnit end = fileUnits.get(windowEnd);

            // Calculate cumulative change for the window
            double cumulativeChange = ((end.getClose() / start.getClose()) - 1) * 100;

            // Check if current row is a spike
            if (cumulativeChange >= spikeThreshold) {
                for (int j = i; j <= windowEnd; j++) {
                    fileUnits.get(j).setTarget(1);
                }
            }
        }

        List<StockUnit> existing = symbolTimelines.getOrDefault(symbol, new ArrayList<>());
        existing.addAll(fileUnits);
        symbolTimelines.put(symbol, existing);

        System.out.println("Loaded " + fileUnits.size() + " entries for " + symbol);

        if (saveData) {
            exportToCSV(fileUnits);
        }
    }

    public static void exportToCSV(List<StockUnit> stocks) {
        try {
            Path filePath = Paths.get(System.getProperty("user.dir"), "rallyMLModel", "highFrequencyStocks.csv");
            File file = filePath.toFile();

            // Check if file exists to determine if we need headers
            boolean fileExists = file.exists();

            // Append mode: true = add to existing file, false = overwrite
            FileWriter csvWriter = new FileWriter(file, true);

            // Write headers only if file is new
            if (!fileExists) {
                csvWriter.append("timestamp,open,high,low,close,volume,target\n");
            }

            // Define the date format for timestamps
            SimpleDateFormat dateFormat = new SimpleDateFormat("EEE MMM dd HH:mm:ss zzz yyyy");
            // Regex pattern to validate the timestamp format (e.g., "Fri Jan 03 14:30:00 GMT 2025")
            String timestampPattern = "[A-Za-z]{3} [A-Za-z]{3} \\d{2} \\d{2}:\\d{2}:\\d{2} [A-Za-z]{3} \\d{4}";

            // Iterate over each StockUnit
            for (StockUnit stock : stocks) {

                // Step 2: Format and validate the timestamp
                try {
                    String timestamp = dateFormat.format(stock.getDateDate());
                    // Step 3: Check if the timestamp matches the expected format
                    if (!timestamp.matches(timestampPattern)) {
                        System.err.println("Warning: Invalid timestamp format: " + timestamp);
                        continue; // Skip this line
                    }

                    // If all checks pass, write the line to the CSV
                    csvWriter.append(escapeCSV(timestamp)).append(",")
                            .append(escapeCSV(String.valueOf(stock.getOpen()))).append(",")
                            .append(escapeCSV(String.valueOf(stock.getHigh()))).append(",")
                            .append(escapeCSV(String.valueOf(stock.getLow()))).append(",")
                            .append(escapeCSV(String.valueOf(stock.getClose()))).append(",")
                            .append(escapeCSV(String.valueOf(stock.getVolume()))).append(",")
                            .append(escapeCSV(String.valueOf(stock.getTarget()))).append("\n");
                } catch (Exception e) {
                    System.out.println("Error formatting stock unit: " + e.getMessage());
                }
            }

            // Flush and close the writer
            csvWriter.flush();
            csvWriter.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static String escapeCSV(String data) {
        if (data == null) {
            return "";  // Handle null values by returning an empty string
        }

        String escapedData = data; // Initialize escapedData with the original data

        // If data contains commas, quotes, or newlines, enclose it in double quotes
        if (data.contains(",") || data.contains("\"") || data.contains("\n")) {
            escapedData = "\"" + data.replace("\"", "\"\"") + "\""; // Replace quotes with escaped quotes and enclose in double quotes
        }
        return escapedData; // Return the escaped data
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
                    e.printStackTrace();
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
            addNotification(currentEvent.getTitle(), currentEvent.getContent(),
                    currentEvent.getTimeSeries(), currentEvent.getLocalDateTime(),
                    currentEvent.getSymbol(), currentEvent.getChange());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void createTimeline(String symbol) {
        try {
            String processedSymbol = symbol.toUpperCase().replace(".TXT", "");
            List<StockUnit> timeline = getSymbolTimeline(processedSymbol);

            if (timeline.isEmpty()) {
                System.out.println("No data available for " + processedSymbol);
                return;
            }

            // Create main price series
            TimeSeries priceSeries = new TimeSeries(processedSymbol + " Price Timeline");
            for (StockUnit unit : timeline) {
                priceSeries.add(new Minute(unit.getDateDate()), unit.getClose());
            }

            // Create datasets
            TimeSeriesCollection priceDataset = new TimeSeriesCollection();
            priceDataset.addSeries(priceSeries);

            // Create indicator series with synchronization
            TimeSeriesCollection indicatorDataset = new TimeSeriesCollection();
            synchronized (indicatorTimeSeries) {
                TimeSeries indicatorSeries = new TimeSeries("Indicator");
                indicatorTimeSeries.getTimePeriods().forEach(period -> {
                    Number value = indicatorTimeSeries.getValue((RegularTimePeriod) period);
                    if (value != null) {
                        indicatorSeries.add((RegularTimePeriod) period, value.doubleValue());
                    }
                });
                indicatorDataset.addSeries(indicatorSeries);
            }

            // Create prediction series with synchronization
            TimeSeriesCollection predictionDataset = new TimeSeriesCollection();
            synchronized (predictionTimeSeries) {
                TimeSeries predictionSeries = new TimeSeries("Prediction");
                predictionTimeSeries.getTimePeriods().forEach(period -> {
                    Number value = predictionTimeSeries.getValue((RegularTimePeriod) period);
                    if (value != null) {
                        predictionSeries.add((RegularTimePeriod) period, value.doubleValue());
                    }
                });
                predictionDataset.addSeries(predictionSeries);
            }

            // Create chart
            JFreeChart chart = ChartFactory.createTimeSeriesChart(
                    processedSymbol + " Analysis",
                    "Time",
                    "Price",
                    priceDataset,
                    true,
                    true,
                    false
            );

            XYPlot plot = getXyPlot(chart);

            // Add datasets to plot
            plot.setDataset(1, indicatorDataset);
            plot.setDataset(2, predictionDataset);

            // Map datasets to axes
            plot.mapDatasetToRangeAxis(1, 1);  // Indicator to left axis
            plot.mapDatasetToRangeAxis(2, 2);  // Prediction to right axis

            // Configure renderers
            // Indicator renderer (subtle blue line without markers)
            XYLineAndShapeRenderer indicatorRenderer = new XYLineAndShapeRenderer();
            indicatorRenderer.setSeriesPaint(0, new Color(100, 100, 255));
            indicatorRenderer.setSeriesStroke(0, new BasicStroke(1f));
            indicatorRenderer.setSeriesShapesVisible(0, false);  // Disable shapes (dots, arrows, etc.)

            // Prediction renderer (bright red line without markers)
            XYLineAndShapeRenderer predictionRenderer = new XYLineAndShapeRenderer();
            predictionRenderer.setSeriesPaint(0, new Color(78, 255, 0));
            predictionRenderer.setSeriesStroke(0, new BasicStroke(1f));
            predictionRenderer.setSeriesShapesVisible(0, false);  // Disable shapes

            // Assign renderers to datasets
            plot.setRenderer(1, indicatorRenderer);
            plot.setRenderer(2, predictionRenderer);

            // Configure main series renderer (solid black line without markers)
            XYLineAndShapeRenderer priceRenderer = (XYLineAndShapeRenderer) plot.getRenderer();
            priceRenderer.setSeriesPaint(0, Color.BLACK);  // Solid black line
            priceRenderer.setSeriesStroke(0, new BasicStroke(0.5f));
            priceRenderer.setSeriesShapesVisible(0, false);  // Disable shapes

            // Add notification markers
            addNotificationMarkers(plot, processedSymbol, timeline);

            // Chart panel and frame setup
            ChartPanel chartPanel = createChartPanel(chart);
            JPanel controlPanel = getjPanel(chartPanel);

            JFrame frame = new JFrame("Price Timeline Analysis");
            frame.setLayout(new BorderLayout());
            frame.add(chartPanel, BorderLayout.CENTER);
            frame.add(controlPanel, BorderLayout.SOUTH);
            frame.pack();
            frame.setSize(1700, 1000);
            frame.setLocation(60, 20);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setVisible(true);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @NotNull
    private static XYPlot getXyPlot(JFreeChart chart) {
        XYPlot plot = chart.getXYPlot();

        // Configure secondary axes
        // Indicator Axis (left side)
        NumberAxis indicatorAxis = new NumberAxis("Indicator");
        indicatorAxis.setRange(-1.5, 1.5);
        indicatorAxis.setAxisLinePaint(Color.BLUE);
        plot.setRangeAxis(1, indicatorAxis);

        // Prediction Axis (right side)
        NumberAxis predictionAxis = new NumberAxis("Prediction");
        predictionAxis.setRange(0, 1);
        predictionAxis.setAxisLinePaint(Color.RED);
        plot.setRangeAxis(2, predictionAxis);
        return plot;
    }

    private static ChartPanel createChartPanel(JFreeChart chart) {
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setMouseWheelEnabled(true);
        chartPanel.setZoomAroundAnchor(true);
        chartPanel.setRangeZoomable(true);
        chartPanel.setDomainZoomable(true);

        chartPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Point2D p = chartPanel.translateScreenToJava2D(e.getPoint());
                XYPlot plot = chart.getXYPlot();
                ValueAxis xAxis = plot.getDomainAxis();
                ValueAxis yAxis = plot.getRangeAxis();

                double x = xAxis.java2DToValue(p.getX(), chartPanel.getScreenDataArea(), plot.getDomainAxisEdge());
                double y = yAxis.java2DToValue(p.getY(), chartPanel.getScreenDataArea(), plot.getRangeAxisEdge());

                if (Double.isNaN(point1X)) {
                    point1X = x;
                    point1Y = y;
                    addFirstMarker(plot, point1X);
                } else {
                    point2X = x;
                    point2Y = y;
                    addSecondMarkerAndShade(plot);
                    resetPoints();
                }
            }
        });
        return chartPanel;
    }

    private static void addNotificationMarkers(XYPlot plot, String symbol, List<StockUnit> timeline) {
        for (Notification notification : notificationsForPLAnalysis) {
            if (!notification.getSymbol().equalsIgnoreCase(symbol)) continue;

            LocalDateTime notifyTime = notification.getLocalDateTime();
            Integer index = getIndexForTime(symbol, notifyTime);

            if (index != null && index < timeline.size()) {
                StockUnit unit = timeline.get(index);
                ValueMarker marker = new ValueMarker(unit.getDateDate().getTime());
                marker.setPaint(new Color(220, 20, 60, 150));
                marker.setStroke(new BasicStroke(1.5f));
                plot.addDomainMarker(marker);
            }
        }
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

    private static void resetPoints() {
        point1X = Double.NaN;
        point1Y = Double.NaN;
        point2X = Double.NaN;
        point2Y = Double.NaN;
    }

    private static void getNext5Minutes(double capital, LocalDateTime startTime, String symbol) {
        symbol = symbol.toUpperCase();
        List<StockUnit> timeline = symbolTimelines.getOrDefault(symbol, Collections.emptyList());

        if (timeline.isEmpty()) {
            System.out.println("No data available for " + symbol);
            return;
        }

        Integer startIndex = symbolTimeIndex.getOrDefault(symbol, Collections.emptyMap()).get(startTime);

        if (startIndex == null || startIndex < 0 || startIndex >= timeline.size()) {
            System.out.println("Start time not found in data: " + startTime);
            return;
        }

        System.out.printf("\u001B[34mSimulating %s from %s with $%.2f\u001B[0m%n",
                symbol, startTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME), capital);

        double simulatedCapital = capital;
        int predictionsMade = 0;
        final int maxSteps = Math.min(5, timeline.size() - startIndex - 1);

        for (int i = 1; i <= maxSteps; i++) {
            StockUnit futureUnit = timeline.get(startIndex + i);
            double change = futureUnit.getPercentageChange();
            simulatedCapital *= (1 + (change / 100));
            predictionsMade++;

            System.out.printf("\u001B[33m%d min later: %+.2f%% on %s → $%.2f\u001B[0m%n",
                    i, change,
                    futureUnit.getLocalDateTimeDate().format(DateTimeFormatter.ISO_LOCAL_TIME),
                    simulatedCapital
            );
        }

        System.out.printf("\u001B[32mFinal simulation result: $%.2f (%.2f%% change)\u001B[0m%n",
                simulatedCapital,
                ((simulatedCapital - capital) / capital) * 100
        );

        if (predictionsMade < 5) {
            System.out.println("Warning: Only " + predictionsMade + " predictions available");
        }
    }

    // Record for trade results
    private record TradeResult(double newCapital, LocalDateTime lastTradeTime) {
    }
}