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
import org.jfree.chart.renderer.xy.CandlestickRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.time.Second;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;
import org.jfree.data.xy.DefaultOHLCDataset;
import org.jfree.data.xy.OHLCDataItem;
import org.jfree.data.xy.OHLCDataset;

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
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import static org.crecker.dataTester.getData;
import static org.crecker.dataTester.parseStockUnit;
import static org.crecker.mainDataHandler.*;
import static org.crecker.mainUI.*;

public class pLTester {
    // Index map for quick timestamp lookups
    private static final Map<String, Map<LocalDateTime, Integer>> symbolTimeIndex = new ConcurrentHashMap<>();
    static JLabel percentageChange;
    static List<TimeInterval> labeledIntervals = new ArrayList<>();
    private static double point1X = Double.NaN;
    private static double point1Y = Double.NaN;
    private static double point2X = Double.NaN;
    private static double point2Y = Double.NaN;
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;

    // Controls over trading
    public final static String[] SYMBOLS = {"QBTS"};
    private final static boolean useCandles = true;
    private final static int cut = 30000;

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
        // ANSI colour codes for terminal output
        final String RESET = "\u001B[0m";
        final String RED = "\u001B[31m";
        final String GREEN = "\u001B[32m";
        final String YELLOW = "\u001B[33m";
        final String CYAN = "\u001B[36m";
        final String WHITE_BOLD = "\u001B[1;37m";
        mainUI.useCandles = useCandles;

        double INITIAL_CAPITAL = 100000;
        prepData();

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
        createSuperChart(SYMBOLS[0]);

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

            System.out.println(WHITE_BOLD + "\n=== NEW TRADE OPPORTUNITY ===" + RESET);
            System.out.printf(YELLOW + "Notification Time: %s%n" + RESET, notifyTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            notification.showNotification();

            Integer baseIndex = getIndexForTime(symbol, notifyTime);

            if (baseIndex == null || baseIndex >= timeline.size() - 5) {
                System.out.println(RED + "Invalid index for trading - insufficient data" + RESET);
                continue;
            }

            int totalMinutes = 5;
            int offset = 0;
            while (offset < totalMinutes) {
                int currentIndex = baseIndex + offset;
                StockUnit unit = timeline.get(currentIndex);
                lastProcessedEndTime = timeline.get(currentIndex).getLocalDateTimeDate();

                System.out.printf(CYAN + "\nMinute %d/%d: %s | Price: %.3f | Change: %.3f%% | Symbol: %s%n" + RESET,
                        offset + 1, totalMinutes,
                        unit.getLocalDateTimeDate().format(DateTimeFormatter.ISO_LOCAL_TIME),
                        unit.getClose(),
                        unit.getPercentageChange(),
                        symbol);

                notification.addDataPoint(unit);

                System.out.print(YELLOW + "Enter trade? (y/n/l/exit): " + RESET);
                String input = scanner.nextLine().trim().toLowerCase();

                if (input.equals("y")) {
                    // ENTER TRADE
                    tradeEntryCapital = capital;
                    tradeEntryTime = unit.getLocalDateTimeDate();
                    tradeEntryIndex = currentIndex;
                    inTrade = true;
                    double totalChange = 0.0;
                    successfulCalls++;
                    System.out.printf(GREEN + "\nENTERED TRADE AT %s WITH €%.2f%n" + RESET,
                            tradeEntryTime.format(DateTimeFormatter.ISO_LOCAL_TIME),
                            tradeEntryCapital);

                    // PROCESS MINUTES UNTIL EXIT
                    for (int i = tradeEntryIndex + 1; i < timeline.size(); i++) {
                        StockUnit minuteUnit = timeline.get(i);
                        totalChange += minuteUnit.getPercentageChange();
                        System.out.printf(CYAN + "\n[TRADE UPDATE] %s | Price: %.3f | Change: %.3f%% | Total Change %.3f%%%n" + RESET,
                                minuteUnit.getLocalDateTimeDate().format(DateTimeFormatter.ISO_LOCAL_TIME),
                                minuteUnit.getClose(),
                                minuteUnit.getPercentageChange(),
                                totalChange);

                        notification.addDataPoint(minuteUnit);

                        System.out.print(RED + "Exit trade now? (y/n): " + RESET);
                        String exitChoice = scanner.nextLine().trim().toLowerCase();

                        if (exitChoice.equals("y")) {
                            capital = calculateTradeValue(timeline, tradeEntryIndex + 1, i, tradeEntryCapital);
                            inTrade = false;
                            System.out.printf(GREEN + "\nEXITED TRADE AT %s | NEW CAPITAL: €%.2f%n" + RESET,
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
                        System.out.printf(RED + "\n[AUTO-CLOSE] FINAL CAPITAL: €%.2f%n" + RESET, capital);
                    }
                    break; // Exit the while loop after entering a trade.
                } else if (input.equalsIgnoreCase("exit")) {
                    earlyStop = true;
                    break;
                } else if (input.equals("l")) {
                    totalMinutes += 5; // Extend the trade entry window by 5 minutes.
                    System.out.println(YELLOW + "Extending the trade entry window by 5 minutes..." + RESET);
                }
                offset++;
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

    private static void prepData() {
        File cacheDir = new File(CACHE_DIR);
        if (!cacheDir.exists()) cacheDir.mkdirs();

        // Calculation of rallies, Process data for each file
        Arrays.stream(pLTester.SYMBOLS).forEach(symbol -> {
            try {
                String fileName = symbol + ".txt";
                String cachePath = Paths.get(CACHE_DIR, fileName).toString();
                processStockDataFromFile(cachePath, symbol, pLTester.cut);
            } catch (IOException e) {
                e.printStackTrace();
            }
        });

        synchronized (symbolTimelines) {
            symbolTimelines.forEach((symbol, timeline) -> {
                if (timeline.size() < 2) {
                    logTextArea.append("Not enough data for " + symbol + "\n");
                    return;
                }

                for (int i = 1; i < timeline.size(); i++) {
                    StockUnit current = timeline.get(i);
                    StockUnit previous = timeline.get(i - 1);

                    if (previous.getClose() > 0) {
                        double change = ((current.getClose() - previous.getClose()) / previous.getClose()) * 100;
                        change = Math.abs(change) >= 14 ? previous.getPercentageChange() : change;
                        current.setPercentageChange(change);
                    }
                }
            });
        }

        precomputeIndicatorRanges(false);
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

            // Iterate over each StockUnit
            for (StockUnit stock : stocks) {

                // Step 2: Format and validate the timestamp
                try {
                    // If all checks pass, write the line to the CSV
                    csvWriter.append(escapeCSV(stock.getDateDate().toString())).append(",")
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
                    currentEvent.getStockUnitList(), currentEvent.getLocalDateTime(),
                    currentEvent.getSymbol(), currentEvent.getChange(), currentEvent.getConfig());
        } catch (Exception e) {
            e.printStackTrace();
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

            JFreeChart chart;
            XYPlot plot;

            if (pLTester.useCandles) {
                // Create OHLC dataset for candlesticks
                OHLCDataItem[] dataItems = new OHLCDataItem[timeline.size()];
                for (int i = 0; i < timeline.size(); i++) {
                    StockUnit unit = timeline.get(i);
                    dataItems[i] = new OHLCDataItem(
                            unit.getDateDate(),
                            unit.getOpen(),
                            unit.getHigh(),
                            unit.getLow(),
                            unit.getClose(),
                            unit.getVolume()
                    );
                }

                OHLCDataset ohlcDataset = new DefaultOHLCDataset(
                        processedSymbol + " Candles",
                        dataItems
                );

                // Create chart with OHLC dataset
                chart = ChartFactory.createCandlestickChart(
                        processedSymbol + " Candlestick Chart",
                        "Time",
                        "Price",
                        ohlcDataset,
                        true
                );

                // Get reference to the plot
                plot = chart.getXYPlot();

                // Set auto-range for axes and exclude zero from the range
                plot.getDomainAxis().setAutoRange(true);
                NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
                rangeAxis.setAutoRange(true);
                rangeAxis.setAutoRangeIncludesZero(false); // Exclude zero from auto-range
                rangeAxis.configure(); // Force recalculation of the axis range

                // Configure candlestick renderer
                CandlestickRenderer renderer = new CandlestickRenderer();
                renderer.setAutoWidthMethod(CandlestickRenderer.WIDTHMETHOD_SMALLEST);
                renderer.setUpPaint(Color.GREEN);    // Color for "up" candles (close >= open)
                renderer.setDownPaint(Color.RED);    // Color for "down" candles (close < open)
                renderer.setUseOutlinePaint(true);
                renderer.setDrawVolume(true);
                plot.setRenderer(renderer);

            } else {
                // Create main price series
                TimeSeries closeSeries = new TimeSeries(processedSymbol + " Close");
                TimeSeries openSeries = new TimeSeries(processedSymbol + " Open");
                TimeSeries highSeries = new TimeSeries(processedSymbol + " High");
                TimeSeries lowSeries = new TimeSeries(processedSymbol + " Low");

                for (StockUnit unit : timeline) {
                    closeSeries.add(new Second(unit.getDateDate()), unit.getClose());
                    openSeries.add(new Second(unit.getDateDate()), unit.getOpen());
                    lowSeries.add(new Second(unit.getDateDate()), unit.getLow());
                    highSeries.add(new Second(unit.getDateDate()), unit.getHigh());
                }

                // Create datasets
                TimeSeriesCollection priceDataset = new TimeSeriesCollection();
                priceDataset.addSeries(closeSeries); // Index 0
                priceDataset.addSeries(highSeries);  // Index 1
                priceDataset.addSeries(lowSeries);   // Index 2
                priceDataset.addSeries(openSeries);  // Index 3

                // Create chart
                chart = ChartFactory.createTimeSeriesChart(
                        processedSymbol + " Analysis",
                        "Time",
                        "Price",
                        priceDataset,
                        true,
                        true,
                        false
                );

                plot = chart.getXYPlot();

                // Configure main series renderer (solid black line without markers)
                XYLineAndShapeRenderer renderer = (XYLineAndShapeRenderer) plot.getRenderer();

                // Close - Black
                renderer.setSeriesPaint(0, Color.BLACK);
                renderer.setSeriesStroke(0, new BasicStroke(1f));
                renderer.setSeriesShapesVisible(0, false);

                // High - Red
                renderer.setSeriesPaint(1, Color.RED);
                renderer.setSeriesStroke(1, new BasicStroke(1f));
                renderer.setSeriesShapesVisible(1, false);

                // Low - Blue
                renderer.setSeriesPaint(2, Color.BLUE);
                renderer.setSeriesStroke(2, new BasicStroke(1f));
                renderer.setSeriesShapesVisible(2, false);

                // Open - Green
                renderer.setSeriesPaint(3, Color.GREEN);
                renderer.setSeriesStroke(3, new BasicStroke(1f));
                renderer.setSeriesShapesVisible(3, false);
            }

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

        timeline.stream()
                .filter(unit -> {
                    long unitTime = unit.getDateDate().getTime();
                    return unitTime >= startTime && unitTime <= endTime;
                })
                .forEach(unit -> unit.setTarget(value));
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