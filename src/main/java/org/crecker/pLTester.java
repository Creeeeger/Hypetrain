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
import org.jfree.chart.ui.Layer;
import org.jfree.data.Range;
import org.jfree.data.time.RegularTimePeriod;
import org.jfree.data.time.Second;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
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

import static org.crecker.dataTester.getData;
import static org.crecker.dataTester.parseStockUnit;
import static org.crecker.mainDataHandler.*;
import static org.crecker.mainUI.addNotification;
import static org.crecker.mainUI.gui;

public class pLTester {
    // Index map for quick timestamp lookups
    private static final Map<String, Map<LocalDateTime, Integer>> symbolTimeIndex = new ConcurrentHashMap<>();
    static JLabel percentageChange;
    static List<TimeInterval> labeledIntervals = new ArrayList<>();
    static boolean tradeView = true;
    static int feature = 0;
    private static double point1X = Double.NaN;
    private static double point1Y = Double.NaN;
    private static double point2X = Double.NaN;
    private static double point2Y = Double.NaN;
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;
    private static boolean isAdjusting = false;

    public static void main(String[] args) {
        //updateStocks();
        PLAnalysis();
    }

    private static void updateStocks() {
        // Add / remove stock which should get added / updated
        for (String stock : Arrays.asList("SMCI", "IONQ", "WOLF", "MARA", "NVDA", "QBTS", "IREN", "PLTR", "MSTR", "ARM")) {
            getData(stock);
        }
    }

    public static void PLAnalysis() {
        final String[] SYMBOLS = {"PLTR.txt"};
        double INITIAL_CAPITAL = 130000;
        final int FEE = 0;
        int cut = 900;
        prepData(SYMBOLS, cut);

        // Preprocess indices during data loading
        Arrays.stream(SYMBOLS).forEach(symbol -> buildTimeIndex(symbol.replace(".txt", ""),
                getSymbolTimeline(symbol.replace(".txt", ""))));

        double capital = INITIAL_CAPITAL;
        int successfulCalls = 0;

        // Trade state tracking
        boolean inTrade;
        boolean earlyStop = false;
        LocalDateTime tradeEntryTime;
        double tradeEntryCapital;
        int tradeEntryIndex;

        Scanner scanner = new Scanner(System.in);
        LocalDateTime lastProcessedEndTime = null;

        if (tradeView) {
            createTimeline(SYMBOLS[0]);
            createSuperChart(SYMBOLS[0]);
        }

        // Cache timelines per notification
        Map<String, List<StockUnit>> timelineCache = new HashMap<>();

        for (Notification notification : notificationsForPLAnalysis) {
            LocalDateTime notifyTime = notification.getLocalDateTime();

            if (lastProcessedEndTime != null && !notifyTime.isAfter(lastProcessedEndTime)) {
                continue;
            }

            if (gui != null) createNotification(notification);

            String symbol = notification.getSymbol();
            List<StockUnit> timeline = timelineCache.computeIfAbsent(symbol, mainDataHandler::getSymbolTimeline);

            System.out.println("\n=== NEW TRADE OPPORTUNITY ===");
            System.out.printf("Notification Time: %s%n", notifyTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));

            if (!tradeView) {
                notification.showNotification();
            }

            Integer baseIndex = getIndexForTime(symbol, notifyTime);

            if (baseIndex == null || baseIndex >= timeline.size() - 5) {
                System.out.println("Invalid index for trading - insufficient data");
                continue;
            }

            lastProcessedEndTime = timeline.get(baseIndex + 5).getLocalDateTimeDate();

            for (int offset = 0; offset <= 4; offset++) {
                int currentIndex = baseIndex + offset;
                StockUnit unit = timeline.get(currentIndex);

                System.out.printf("\nMinute %d/%d: %s | Price: %.3f | Change: %.3f%% | Symbol: %s%n",
                        offset + 1, 5,
                        unit.getLocalDateTimeDate().format(DateTimeFormatter.ISO_LOCAL_TIME),
                        unit.getClose(),
                        unit.getPercentageChange(),
                        symbol);

                notification.addDataPoint(unit);

                System.out.print("Enter trade? (y/n/exit): ");
                String input = scanner.nextLine().trim().toLowerCase();

                if (input.equals("y")) {
                    // ENTER TRADE
                    tradeEntryCapital = capital;
                    tradeEntryTime = unit.getLocalDateTimeDate();
                    tradeEntryIndex = currentIndex;
                    inTrade = true;
                    double totalChange = 0.0;
                    successfulCalls++;
                    System.out.printf("\nENTERED TRADE AT %s WITH €%.2f%n",
                            tradeEntryTime.format(DateTimeFormatter.ISO_LOCAL_TIME),
                            tradeEntryCapital);

                    // PROCESS MINUTES UNTIL EXIT
                    for (int i = tradeEntryIndex + 1; i < timeline.size(); i++) {
                        StockUnit minuteUnit = timeline.get(i);
                        totalChange += minuteUnit.getPercentageChange();
                        System.out.printf("\n[TRADE UPDATE] %s | Price: %.3f | Change: %.3f%% | Total Change %.3f%%%n",
                                minuteUnit.getLocalDateTimeDate().format(DateTimeFormatter.ISO_LOCAL_TIME),
                                minuteUnit.getClose(),
                                minuteUnit.getPercentageChange(),
                                totalChange);

                        notification.addDataPoint(minuteUnit);

                        System.out.print("Exit trade now? (y/n): ");
                        String exitChoice = scanner.nextLine().trim().toLowerCase();

                        if (exitChoice.equals("y")) {
                            capital = calculateTradeValue(timeline, tradeEntryIndex + 1, i, tradeEntryCapital);
                            capital -= FEE;
                            inTrade = false;
                            System.out.printf("\nEXITED TRADE AT %s | NEW CAPITAL: €%.2f%n",
                                    minuteUnit.getLocalDateTimeDate().format(DateTimeFormatter.ISO_LOCAL_TIME),
                                    capital);

                            lastProcessedEndTime = minuteUnit.getLocalDateTimeDate();
                            break;
                        }
                    }

                    // AUTO-CLOSE IF STILL IN TRADE
                    if (inTrade) {
                        int finalIndex = timeline.size() - 1;
                        capital = calculateTradeValue(timeline, tradeEntryIndex + 1, finalIndex, tradeEntryCapital);
                        capital -= FEE;
                        System.out.printf("\n[AUTO-CLOSE] FINAL CAPITAL: €%.2f%n", capital);
                    }
                    break;
                }

                if (input.equalsIgnoreCase("exit")) {
                    earlyStop = true;
                    break;
                }
            }

            if (earlyStop) {
                break;
            }

            notification.closeNotification();
        }

        logFinalResults(capital, INITIAL_CAPITAL, successfulCalls);

        scanner.close();
    }

    private static double calculateTradeValue(List<StockUnit> timeline, int entryIndex, int exitIndex, double capital) {
        double cumulative = 1.0;
        for (int i = entryIndex; i <= exitIndex; i++) {
            cumulative *= (1 + timeline.get(i).getPercentageChange() / 100);
        }

        return capital * cumulative;
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

    private static void logFinalResults(double capital, double initial, int calls) {
        double revenue;

        // Ensure losses aren't made cheaper by tax reduction
        if ((capital - initial) > 0) {
            revenue = (capital - initial) * 0.75;
        } else {
            revenue = (capital - initial);
        }

        if (calls > 0) {
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

        dataTester.calculateStockPercentageChange();
        calculateSpikesInRally(frameSize, false);
    }

    public static void processStockDataFromFile(String filePath, String symbol, int retainLast) throws IOException {
        List<StockUnit> fileUnits = readStockUnitsFromFile(filePath, retainLast);
        symbol = symbol.toUpperCase();

        fileUnits.forEach(unit -> unit.setTarget(0));

        List<StockUnit> existing = symbolTimelines.getOrDefault(symbol, new ArrayList<>());
        existing.addAll(fileUnits);
        symbolTimelines.put(symbol, existing);
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
                    currentEvent.getSymbol(), currentEvent.getChange(), currentEvent.getConfig());
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
            TimeSeries priceSeries = new TimeSeries(processedSymbol + " Price");
            for (StockUnit unit : timeline) {
                priceSeries.add(new Second(unit.getDateDate()), unit.getClose());
            }
            TimeSeriesCollection priceDataset = new TimeSeriesCollection();
            priceDataset.addSeries(priceSeries);

            // Create prediction dataset
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

            // Prepare frame for multiple charts
            JFrame frame = new JFrame("Multi-Feature Analysis");
            frame.setLayout(new GridLayout(4, 2)); // 4 rows, 2 columns (7 charts)
            List<ChartPanel> chartPanels = new ArrayList<>();

            // Create a chart for each feature (0,1,2,4,5,6,7)
            int[] featureIndices = {0, 1, 2, 4, 5, 6, 7};
            for (int i : featureIndices) {
                TimeSeries featureSeries = featureTimeSeriesArray[i];
                TimeSeriesCollection featureDataset = new TimeSeriesCollection();
                featureDataset.addSeries(featureSeries);

                double lower = featureSeries.getMinY() - Math.abs(featureSeries.getMinY() * 0.05);
                if (lower == 0.0) {
                    lower = -0.05;
                }

                // Create chart
                JFreeChart chart = createFeatureChart(
                        processedSymbol,
                        priceDataset,
                        predictionDataset,
                        featureDataset,
                        i,
                        timeline,
                        lower,
                        featureSeries.getMaxY() + Math.abs(featureSeries.getMaxY() * 0.05)
                );

                ChartPanel chartPanel = new ChartPanel(chart);
                chartPanel.setDomainZoomable(true); // Allow domain zoom
                chartPanel.setRangeZoomable(false); // Disable range zoom
                createPercentageMarkers(chart, chartPanel);

                chartPanels.add(chartPanel);
                frame.add(chartPanel);
            }

            JPanel chartContainer = new JPanel(new BorderLayout());

            // Create percentage label with proper positioning
            percentageChange = new JLabel("Percentage Change");

            // Add label below title but above chart
            chartContainer.add(percentageChange, BorderLayout.NORTH);
            frame.add(chartContainer);

            // Link domain axes for synchronized zooming
            linkDomainAxes(chartPanels);

            // Configure frame
            frame.pack();
            frame.setSize(1700, 1000);
            frame.setLocation(60, 20);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.setVisible(true);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static JFreeChart createFeatureChart(String symbol, TimeSeriesCollection priceDataset, TimeSeriesCollection predictionDataset,
                                                 TimeSeriesCollection featureDataset, int featureIndex, List<StockUnit> timeline, double lower, double upper) {

        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                symbol + " - Feature " + featureIndex,
                "Time",
                "Value",
                priceDataset,
                true,
                true,
                false
        );

        XYPlot plot = chart.getXYPlot();

        // Clear existing axes
        plot.clearRangeAxes();

        // Create and configure axes
        // 1. Price axis (primary - left)
        NumberAxis priceAxis = new NumberAxis("Price");
        priceAxis.setAutoRangeIncludesZero(false);
        plot.setRangeAxis(0, priceAxis);

        // 2. Feature axis (secondary - left)
        NumberAxis featureAxis = new NumberAxis("Feature " + featureIndex);
        featureAxis.setAutoRangeIncludesZero(false);
        plot.setRangeAxis(1, featureAxis);

        // 3. Prediction axis (tertiary - right)
        NumberAxis predictionAxis = new NumberAxis("Prediction");
        predictionAxis.setAutoRangeIncludesZero(false);
        plot.setRangeAxis(2, predictionAxis);

        // Add datasets with proper axis mapping
        plot.setDataset(0, priceDataset);  // Primary (axis 0)
        plot.setDataset(1, featureDataset); // Secondary (axis 1)
        plot.setDataset(2, predictionDataset); // Tertiary (axis 2)

        plot.mapDatasetToRangeAxis(0, 0);  // Price -> axis 0
        plot.mapDatasetToRangeAxis(1, 1);  // Feature -> axis 1
        plot.mapDatasetToRangeAxis(2, 2);  // Prediction -> axis 2

        // Configure auto-scaling
        priceAxis.setAutoRange(true);
        featureAxis.setAutoRange(false);
        featureAxis.setRange(lower, upper);
        predictionAxis.setRange(-0.05, 1.05);
        predictionAxis.setAutoRange(false);

        // PRESERVE YOUR MARKERS
        // Zero line marker (primary axis)
        ValueMarker zeroMarker = new ValueMarker(0);
        zeroMarker.setPaint(Color.BLACK);
        zeroMarker.setStroke(new BasicStroke(1f));
        plot.addRangeMarker(0, zeroMarker, Layer.FOREGROUND);

        // Notification markers
        addNotificationMarkers(plot, symbol, timeline);

        // Configure renderers
        XYLineAndShapeRenderer priceRenderer = new XYLineAndShapeRenderer();
        priceRenderer.setSeriesPaint(0, Color.BLACK);
        priceRenderer.setSeriesStroke(0, new BasicStroke(1.5f));
        priceRenderer.setSeriesShapesVisible(0, false);

        XYLineAndShapeRenderer featureRenderer = new XYLineAndShapeRenderer();
        featureRenderer.setSeriesPaint(0, new Color(0, 0, 255, 150)); // Blue
        featureRenderer.setSeriesStroke(0, new BasicStroke(1.0f));
        featureRenderer.setSeriesShapesVisible(0, false);

        XYLineAndShapeRenderer predictionRenderer = new XYLineAndShapeRenderer();
        predictionRenderer.setSeriesPaint(0, new Color(255, 165, 0, 200)); // Orange
        predictionRenderer.setSeriesStroke(0, new BasicStroke(1.0f));
        predictionRenderer.setSeriesShapesVisible(0, false);

        plot.setRenderer(0, priceRenderer);
        plot.setRenderer(1, featureRenderer);
        plot.setRenderer(2, predictionRenderer);

        // Styling
        chart.setBackgroundPaint(Color.WHITE);
        plot.setBackgroundPaint(Color.WHITE);
        plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
        plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

        return chart;
    }

    private static void linkDomainAxes(List<ChartPanel> chartPanels) {
        for (ChartPanel panel : chartPanels) {
            ValueAxis domainAxis = panel.getChart().getXYPlot().getDomainAxis();
            domainAxis.addChangeListener(event -> {
                if (!isAdjusting) {
                    isAdjusting = true;
                    Range newDomainRange = domainAxis.getRange();

                    // Update domain for all charts
                    for (ChartPanel otherPanel : chartPanels) {
                        if (otherPanel != panel) {
                            XYPlot otherPlot = otherPanel.getChart().getXYPlot();
                            otherPlot.getDomainAxis().setRange(newDomainRange);
                        }
                    }

                    // Trigger price axis auto-range for all charts
                    for (ChartPanel p : chartPanels) {
                        XYPlot plot = p.getChart().getXYPlot();
                        NumberAxis priceAxis = (NumberAxis) plot.getRangeAxis(0);
                        priceAxis.setAutoRange(true); // Force auto-range
                        priceAxis.configure(); // Recalculate based on visible data
                    }

                    isAdjusting = false;
                }
            });
        }
    }

    private static void createSuperChart(String symbol) {
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
                priceSeries.add(new Second(unit.getDateDate()), unit.getClose());
            }

            // Create datasets
            TimeSeriesCollection priceDataset = new TimeSeriesCollection();
            priceDataset.addSeries(priceSeries);

            // Create indicator series with synchronization
            TimeSeriesCollection indicatorDataset = new TimeSeriesCollection();
            synchronized (featureTimeSeriesArray) {
                TimeSeries indicatorSeries = new TimeSeries("Indicator");
                featureTimeSeriesArray[feature].getTimePeriods().forEach(period -> {
                    Number value = featureTimeSeriesArray[feature].getValue((RegularTimePeriod) period);
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
            // Indicator renderer (distinct blue line without markers)
            XYLineAndShapeRenderer indicatorRenderer = new XYLineAndShapeRenderer();
            indicatorRenderer.setSeriesPaint(0, new Color(0, 0, 255, 150)); // Blue
            indicatorRenderer.setSeriesStroke(0, new BasicStroke(1f));
            indicatorRenderer.setSeriesShapesVisible(0, false);

            // Prediction renderer (orange line without markers)
            XYLineAndShapeRenderer predictionRenderer = new XYLineAndShapeRenderer();
            predictionRenderer.setSeriesPaint(0, new Color(255, 165, 0, 200)); // Orange
            predictionRenderer.setSeriesStroke(0, new BasicStroke(1f));
            predictionRenderer.setSeriesShapesVisible(0, false);

            // Assign renderers to datasets
            plot.setRenderer(1, indicatorRenderer);
            plot.setRenderer(2, predictionRenderer);

            // Create a marker for y = 0
            ValueMarker zeroMarker = new ValueMarker(0);
            zeroMarker.setPaint(Color.BLACK);
            zeroMarker.setStroke(new BasicStroke(1f));

            plot.addRangeMarker(1, zeroMarker, Layer.FOREGROUND);

            // Configure main series renderer (solid black line without markers)
            XYLineAndShapeRenderer priceRenderer = (XYLineAndShapeRenderer) plot.getRenderer();
            priceRenderer.setSeriesPaint(0, Color.BLACK);  // Solid black line
            priceRenderer.setSeriesStroke(0, new BasicStroke(1f));
            priceRenderer.setSeriesShapesVisible(0, false);

            // Set chart background to white for better visibility
            chart.setBackgroundPaint(Color.WHITE);
            plot.setBackgroundPaint(Color.WHITE);
            plot.setDomainGridlinePaint(Color.LIGHT_GRAY);
            plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

            // Add notification markers
            addNotificationMarkers(plot, processedSymbol, timeline);

            // Chart panel and frame setup
            ChartPanel chartPanel = createChartPanel(chart, processedSymbol);
            JPanel controlPanel = getControlPanel(chartPanel, processedSymbol);

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

    private static ChartPanel createChartPanel(JFreeChart chart, String symbol) {
        final Point2D[] dragStartPoint = {null};
        final boolean[] isZoomMode = {false};

        ChartPanel chartPanel = getStockPanel(chart, dragStartPoint);

        createPercentageMarkers(chart, chartPanel);

        // Enable native zoom capabilities
        chartPanel.setDomainZoomable(true);
        chartPanel.setRangeZoomable(true);
        chartPanel.setMouseWheelEnabled(true);

        chartPanel.addMouseListener(new MouseAdapter() {
            private double dragStartTime;

            @Override
            public void mousePressed(MouseEvent e) {
                isZoomMode[0] = e.isShiftDown();

                if (!isZoomMode[0]) {
                    chartPanel.setDomainZoomable(false);
                    chartPanel.setRangeZoomable(false);
                    chartPanel.setMouseWheelEnabled(false);

                    // Start labeling drag
                    dragStartPoint[0] = chartPanel.translateScreenToJava2D(e.getPoint());
                    dragStartTime = chart.getXYPlot().getDomainAxis().java2DToValue(
                            dragStartPoint[0].getX(),
                            chartPanel.getScreenDataArea(),
                            chart.getXYPlot().getDomainAxisEdge()
                    );
                    e.consume();
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                if (!isZoomMode[0] && dragStartPoint[0] != null) {
                    // Handle label completion
                    Point2D endPoint = chartPanel.translateScreenToJava2D(e.getPoint());
                    XYPlot plot = chart.getXYPlot();
                    double endTime = plot.getDomainAxis().java2DToValue(
                            endPoint.getX(),
                            chartPanel.getScreenDataArea(),
                            plot.getDomainAxisEdge()
                    );

                    labeledIntervals.add(new TimeInterval(
                            Math.min(dragStartTime, endTime),
                            Math.max(dragStartTime, endTime)
                    ));

                    updateLabelsForInterval(symbol, dragStartTime, endTime, 1);
                    chartPanel.repaint();
                    e.consume();
                }

                chartPanel.setDomainZoomable(true);
                chartPanel.setRangeZoomable(true);
                chartPanel.setMouseWheelEnabled(true);

                dragStartPoint[0] = null;
                isZoomMode[0] = false;
            }
        });

        chartPanel.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (!isZoomMode[0]) {
                    // Only update preview for labeling
                    chartPanel.repaint();
                    e.consume();
                }
            }
        });

        // Add key listener for shift key state changes
        chartPanel.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_SHIFT) {
                    chartPanel.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
                }
            }

            @Override
            public void keyReleased(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_SHIFT) {
                    chartPanel.setCursor(Cursor.getDefaultCursor());
                }
            }
        });

        // Enable keyboard focus for shift detection
        chartPanel.setFocusable(true);
        chartPanel.requestFocusInWindow();

        return chartPanel;
    }

    @NotNull
    private static ChartPanel getStockPanel(JFreeChart chart, Point2D[] dragStartPoint) {
        return new ChartPanel(chart) {
            @Override
            public void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g.create();

                // Get current plot dimensions
                XYPlot plot = chart.getXYPlot();
                Rectangle2D dataArea = this.getScreenDataArea();
                ValueAxis domainAxis = plot.getDomainAxis();

                // Draw historical intervals
                for (TimeInterval interval : labeledIntervals) {
                    double startX = domainAxis.valueToJava2D(interval.startTime, dataArea, plot.getDomainAxisEdge());
                    double endX = domainAxis.valueToJava2D(interval.endTime, dataArea, plot.getDomainAxisEdge());

                    g2.setColor(new Color(0, 255, 0, 30));
                    g2.fillRect(
                            (int) Math.min(startX, endX),
                            0,
                            (int) Math.abs(endX - startX),
                            this.getHeight()
                    );
                }

                // Draw current drag preview
                if (dragStartPoint[0] != null) {
                    Point2D currentPoint = this.getMousePosition();
                    if (currentPoint != null) {
                        currentPoint = translateScreenToJava2D((Point) currentPoint);
                        double startX = domainAxis.valueToJava2D(
                                domainAxis.java2DToValue(dragStartPoint[0].getX(), dataArea, plot.getDomainAxisEdge()),
                                dataArea, plot.getDomainAxisEdge()
                        );
                        double currentX = domainAxis.valueToJava2D(
                                domainAxis.java2DToValue(currentPoint.getX(), dataArea, plot.getDomainAxisEdge()),
                                dataArea, plot.getDomainAxisEdge()
                        );

                        g2.setColor(new Color(0, 0, 255, 30));
                        g2.fillRect(
                                (int) Math.min(startX, currentX),
                                0,
                                (int) Math.abs(currentX - startX),
                                this.getHeight()
                        );
                    }
                }

                g2.dispose();
            }
        };
    }

    private static void createPercentageMarkers(JFreeChart chart, ChartPanel chartPanel) {
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
    }

    private static JPanel getControlPanel(ChartPanel chartPanel, String symbol) {
        JButton saveButton = new JButton("Save All Labels");
        saveButton.addActionListener(e -> saveLabels(symbol));

        JButton clearButton = new JButton("Clear Last");
        clearButton.addActionListener(e -> clearLastInterval(chartPanel, symbol));

        JButton autoRangeButton = new JButton("Auto Range");
        autoRangeButton.addActionListener(e -> chartPanel.restoreAutoBounds());

        percentageChange = new JLabel("Percentage Change");

        JPanel panel = new JPanel();
        panel.add(autoRangeButton);
        panel.add(saveButton);
        panel.add(clearButton);
        panel.add(percentageChange);

        return panel;
    }

    private static void saveLabels(String symbol) {
        List<StockUnit> timeline = symbolTimelines.get(symbol);
        if (timeline != null) {
            exportToCSV(timeline);
            JOptionPane.showMessageDialog(null, "All labels saved successfully!");
        }
    }

    private static void clearLastInterval(ChartPanel chartPanel, String symbol) {
        if (labeledIntervals.isEmpty()) return;

        TimeInterval last = labeledIntervals.remove(labeledIntervals.size() - 1);
        updateLabelsForInterval(symbol, last.startTime, last.endTime, 0);
        chartPanel.repaint();
    }

    private static void updateLabelsForInterval(String symbol, double startTime, double endTime, int value) {
        List<StockUnit> timeline = symbolTimelines.get(symbol);
        if (timeline == null) return;

        synchronized (featureTimeSeriesArray) {
            timeline.stream()
                    .filter(unit -> {
                        long unitTime = unit.getDateDate().getTime();
                        return unitTime >= startTime && unitTime <= endTime;
                    })
                    .forEach(unit -> {
                        unit.setTarget(value);
                        featureTimeSeriesArray[feature].addOrUpdate(new Second(unit.getDateDate()), value);
                    });
        }
    }

    private static void addNotificationMarkers(XYPlot plot, String symbol, List<StockUnit> timeline) {
        for (Notification notification : notificationsForPLAnalysis) {
            if (!notification.getSymbol().equalsIgnoreCase(symbol)) continue;

            LocalDateTime notifyTime = notification.getLocalDateTime();
            Integer index = getIndexForTime(symbol, notifyTime);

            if (index != null && index < timeline.size()) {
                StockUnit unit = timeline.get(index);
                ValueMarker marker = getValueMarker(notification, unit);
                plot.addDomainMarker(marker);
            }
        }
    }

    @NotNull
    private static ValueMarker getValueMarker(Notification notification, StockUnit unit) {
        ValueMarker marker = new ValueMarker(unit.getDateDate().getTime());
        int config = notification.getConfig();

        Color color;
        if (config == 0) {
            color = new Color(255, 0, 0);         // Bright Red
        } else if (config == 1) {
            color = new Color(255, 140, 0);       // Deep Orange
        } else if (config == 2) {
            color = new Color(0, 128, 255);       // Sky Blue
        } else if (config == 3) {
            color = new Color(34, 177, 76);       // Leaf Green
        } else {
            color = new Color(128, 0, 128);       // Royal Purple
        }

        marker.setPaint(color);
        marker.setStroke(new BasicStroke(1.0f));
        return marker;
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

    private static void resetPoints() {
        point1X = Double.NaN;
        point1Y = Double.NaN;
        point2X = Double.NaN;
        point2Y = Double.NaN;
    }

    private record TimeInterval(double startTime, double endTime) {
    }
}