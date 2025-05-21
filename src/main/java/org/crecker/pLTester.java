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
import static org.crecker.mainUI.addNotification;
import static org.crecker.mainUI.logTextArea;

/**
 * pLTester is the primary entry point and main orchestrator for
 * backtesting and interactive trade simulation on stock data.
 *
 * <p>
 * This class provides methods for:
 * <ul>
 *     <li>Initialising and loading stock data</li>
 *     <li>Interactive step-through of trading notifications</li>
 *     <li>Recording results, capital, and performance metrics</li>
 *     <li>Labeling chart intervals with manual review tools</li>
 * </ul>
 *
 * <p>
 * Key interactive features:
 * <ul>
 *     <li>Displays a chart (candlestick or line) of stock prices</li>
 *     <li>For each notification, allows the user to decide whether to enter a simulated trade</li>
 *     <li>Supports minute-by-minute trade management, capital update, and summary output</li>
 * </ul>
 *
 * @author (your name)
 */
public class pLTester {

    /**
     * Used for quick lookup of data index (list position) by LocalDateTime for each stock symbol.
     * Format: symbolTimeIndex.get(symbol).get(dateTime) = index
     */
    private static final Map<String, Map<LocalDateTime, Integer>> symbolTimeIndex = new ConcurrentHashMap<>();

    /**
     * Label showing current percentage change between selected chart points.
     */
    static JLabel percentageChange;

    /**
     * Stores intervals (start, end) that are manually labeled on the chart for ML or backtest marking.
     */
    static List<TimeInterval> labeledIntervals = new ArrayList<>();

    // Mouse-selected chart coordinates for manual measurement and shading
    private static double point1X = Double.NaN; // Domain value (e.g., timestamp) for first marker
    private static double point1Y = Double.NaN; // Price at first marker
    private static double point2X = Double.NaN; // Domain value for second marker
    private static double point2Y = Double.NaN; // Price at second marker

    // Chart marker graphics
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;

    /**
     * Array of stock symbols currently under analysis/trading.
     */
    public final static String[] SYMBOLS = {"QBTS"};

    /**
     * Whether to display candlestick charts (true) or time series line charts (false).
     */
    private final static boolean useCandles = true;

    /**
     * Maximum number of data rows to use per stock (limits memory, controls recentness of analysis).
     */
    private final static int cut = 7500;

    /**
     * Program entry point.
     * Calls the main analysis routine.
     */
    public static void main(String[] args) {
        // updateStocks(); // Optionally refresh/download all stocks (uncomment if needed)
        PLAnalysis();
    }

    /**
     * Downloads or updates the cached data files for all tracked symbols.
     * This is a batch job, only call when needing fresh raw data.
     */
    private static void updateStocks() {
        // Specify which stocks to update in batch
        for (String stock : Arrays.asList("SMCI", "IONQ", "WOLF", "MARA", "NVDA", "QBTS", "IREN", "PLTR", "MSTR", "ARM")) {
            getData(stock); // Calls external data getter (see dataTester)
        }
    }

    /**
     * Runs the main interactive trade simulation loop.
     * For each notification event, shows chart and steps through a minute-by-minute trade opportunity,
     * letting the user decide when to enter and exit, while tracking capital and statistics.
     * <p>
     * Includes:
     * <ul>
     *     <li>Chart display</li>
     *     <li>Index map preparation for fast data lookups</li>
     *     <li>Iterative trade opportunity review and simulated order entry/exit</li>
     *     <li>Performance output at the end of all runs</li>
     * </ul>
     */
    public static void PLAnalysis() {
        // --- SETUP: ANSI color codes for styled CLI/terminal feedback ---
        final String RESET = "\u001B[0m";
        final String RED = "\u001B[31m";
        final String GREEN = "\u001B[32m";
        final String YELLOW = "\u001B[33m";
        final String CYAN = "\u001B[36m";
        final String WHITE_BOLD = "\u001B[1;37m";

        // --- GLOBAL UI CHART STYLE ---
        mainUI.useCandles = useCandles; // Set global chart mode (candlestick or line)

        // --- TRADING ACCOUNT INIT ---
        double INITIAL_CAPITAL = 160000; // Base starting capital for simulation (can be changed)
        prepData(); // Loads all symbol data, calculates percent changes, prepares all timelines

        // --- FAST INDEX PREP ---
        // For every symbol, build a map from LocalDateTime -> timeline index for ultra-fast lookups during trading
        Arrays.stream(SYMBOLS).forEach(symbol -> buildTimeIndex(
                symbol.replace(".txt", ""),
                getSymbolTimeline(symbol.replace(".txt", ""))));

        // --- LIVE STATS TRACKING ---
        double capital = INITIAL_CAPITAL; // Live capital, updated after every trade
        int successfulCalls = 0;          // Number of successful/entered trades (for statistics)

        // --- PER-TRADE STATE VARS ---
        boolean inTrade;               // True if a trade is currently open
        boolean earlyStop = false;     // True if user opts to end simulation early
        LocalDateTime tradeEntryTime;  // When the most recent trade was entered
        double tradeEntryCapital;      // Capital at trade entry
        int tradeEntryIndex;           // Timeline index where trade started

        // --- INPUT/OUTPUT SETUP ---
        Scanner scanner = new Scanner(System.in);      // CLI scanner for interactive trade decisions
        LocalDateTime lastProcessedEndTime = null;     // To avoid reprocessing overlapping or duplicate events

        // --- UI CHART LAUNCH ---
        createSuperChart(SYMBOLS[0]); // Open the main chart window using first symbol for context

        // --- TIMELINE CACHE ---
        Map<String, List<StockUnit>> timelineCache = new HashMap<>(); // Symbol -> List<StockUnit>, speeds up repeated timeline lookups

        // === MAIN SIMULATION LOOP ===
        for (Notification notification : notificationsForPLAnalysis) {
            LocalDateTime notifyTime = notification.getLocalDateTime();

            // --- DUPLICATE/EARLY FILTER ---
            // Skip this notification if it happens before or overlaps with the last trade exit, avoids redundant or impossible scenarios
            if (lastProcessedEndTime != null && !notifyTime.isAfter(lastProcessedEndTime)) {
                continue;
            }

            // --- GUI NOTIFICATION DISPLAY ---
            // if (gui != null) createNotification(notification); // (If using GUI, pop up the notification event panel) disable Notification requests fist before uncommenting

            // --- GET SYMBOL TIMELINE ---
            String symbol = notification.getSymbol();
            List<StockUnit> timeline = timelineCache.computeIfAbsent(symbol, mainDataHandler::getSymbolTimeline); // Efficiently cache timelines

            // --- TERMINAL OPPORTUNITY DISPLAY ---
            System.out.println(WHITE_BOLD + "\n=== NEW TRADE OPPORTUNITY ===" + RESET);
            System.out.printf(YELLOW + "Notification Time: %s%n" + RESET, notifyTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
            notification.showNotification(); // Print event details

            // --- FIND TRADE ENTRY INDEX ---
            Integer baseIndex = getIndexForTime(symbol, notifyTime); // Get timeline index for event

            // --- DATA SUFFICIENCY CHECK ---
            // Ensure enough bars left for at least 5-minute simulation; skip if data is missing or truncated
            if (baseIndex == null || baseIndex >= timeline.size() - 5) {
                System.out.println(RED + "Invalid index for trading - insufficient data" + RESET);
                continue;
            }

            // --- TRADE ENTRY WINDOW (DEFAULT: 5 candles/minutes, can be extended) ---
            int totalMinutes = 5;
            int offset = 0;
            while (offset < totalMinutes) {
                int currentIndex = baseIndex + offset;
                StockUnit unit = timeline.get(currentIndex);
                lastProcessedEndTime = timeline.get(currentIndex).getLocalDateTimeDate(); // Update to prevent overlap in future loops

                // --- PRINT CURRENT CANDLE DETAILS ---
                System.out.printf(CYAN + "\nMinute %d/%d: %s | Price: %.3f | Change: %.3f%% | Symbol: %s%n" + RESET,
                        offset + 1, totalMinutes,
                        unit.getLocalDateTimeDate().format(DateTimeFormatter.ISO_LOCAL_TIME),
                        unit.getClose(),
                        unit.getPercentageChange(),
                        symbol);

                // --- ADD TO NOTIFICATION DATA (FOR CHARTS/EXPORT) ---
                notification.addDataPoint(unit);

                // --- TRADE ENTRY PROMPT (INTERACTIVE DECISION) ---
                System.out.print(YELLOW + "Enter trade? (y/n/l/exit): " + RESET);
                String input = scanner.nextLine().trim().toLowerCase();

                if (input.equals("y")) {
                    // === TRADE ENTRY LOGIC ===
                    tradeEntryCapital = capital;                        // Store current capital at entry
                    tradeEntryTime = unit.getLocalDateTimeDate();       // Store entry time
                    tradeEntryIndex = currentIndex;                     // Store entry index
                    inTrade = true;                                     // Mark that we are now in a trade
                    double totalChange = 0.0;                           // Running change % for trade stats
                    successfulCalls++;                                  // Count this as a completed trade

                    System.out.printf(GREEN + "\nENTERED TRADE AT %s WITH €%.2f%n" + RESET,
                            tradeEntryTime.format(DateTimeFormatter.ISO_LOCAL_TIME),
                            tradeEntryCapital);

                    // --- TRADE MANAGEMENT LOOP (USER CAN EXIT ANY MINUTE) ---
                    for (int i = tradeEntryIndex + 1; i < timeline.size(); i++) {
                        StockUnit minuteUnit = timeline.get(i);
                        totalChange += minuteUnit.getPercentageChange();
                        System.out.printf(CYAN + "\n[TRADE UPDATE] %s | Price: %.3f | Change: %.3f%% | Total Change %.3f%%%n" + RESET,
                                minuteUnit.getLocalDateTimeDate().format(DateTimeFormatter.ISO_LOCAL_TIME),
                                minuteUnit.getClose(),
                                minuteUnit.getPercentageChange(),
                                totalChange);

                        // Add to notification for visualization/tracking
                        notification.addDataPoint(minuteUnit);

                        // --- TRADE EXIT PROMPT ---
                        System.out.print(RED + "Exit trade now? (y/n): " + RESET);
                        String exitChoice = scanner.nextLine().trim().toLowerCase();

                        if (exitChoice.equals("y")) {
                            // === TRADE EXIT LOGIC ===
                            capital = calculateTradeValue(timeline, tradeEntryIndex + 1, i, tradeEntryCapital); // Update capital with new result
                            inTrade = false;
                            System.out.printf(GREEN + "\nEXITED TRADE AT %s | NEW CAPITAL: €%.2f%n" + RESET,
                                    minuteUnit.getLocalDateTimeDate().format(DateTimeFormatter.ISO_LOCAL_TIME),
                                    capital);

                            lastProcessedEndTime = minuteUnit.getLocalDateTimeDate(); // Don't revisit candles
                            break; // Exit trade management loop
                        }
                    }

                    // --- AUTO-CLOSE AT END OF DATA IF NEVER MANUALLY EXITED ---
                    if (inTrade) {
                        int finalIndex = timeline.size() - 1;
                        capital = calculateTradeValue(timeline, tradeEntryIndex + 1, finalIndex, tradeEntryCapital);
                        System.out.printf(RED + "\n[AUTO-CLOSE] FINAL CAPITAL: €%.2f%n" + RESET, capital);
                    }
                    break; // Break out of entry window, move to next notification
                } else if (input.equalsIgnoreCase("exit")) {
                    // --- USER ENDS ENTIRE SIMULATION EARLY ---
                    earlyStop = true;
                    break;
                } else if (input.equals("l")) {
                    // --- EXTEND ENTRY WINDOW (GIVES USER MORE TIME TO DECIDE) ---
                    totalMinutes += 5; // Add another 5 candles/minutes to entry window
                    System.out.println(YELLOW + "Extending the trade entry window by 5 minutes..." + RESET);
                }
                offset++; // Advance to next minute/candle in entry window
            }

            // --- GLOBAL EARLY STOP LOGIC ---
            if (earlyStop) {
                break; // Exit main notification loop as well
            }

            // --- CLEAN UP UI NOTIFICATION PANEL/WINDOW ---
            notification.closeNotification();
        }

        // --- FINAL RESULTS/OUTPUT SECTION ---
        logFinalResults(capital, INITIAL_CAPITAL, successfulCalls); // Output summary, win %, net gain/loss, etc.

        // --- RESOURCE CLEANUP ---
        scanner.close(); // Always close scanner to free system resources
    }

    /**
     * Calculates the ending capital of a trade given a start and end index in the timeline.
     * Multiplies the initial capital by the compounded percentage change over each bar in the trade.
     *
     * @param timeline   List of StockUnit objects representing price data.
     * @param entryIndex Index of the trade entry (exclusive: starts applying from entryIndex).
     * @param exitIndex  Index of the trade exit (inclusive).
     * @param capital    Capital at entry (in euros, dollars, etc.).
     * @return Final capital after applying compounded changes over the interval.
     */
    private static double calculateTradeValue(List<StockUnit> timeline, int entryIndex, int exitIndex, double capital) {
        // Initialize cumulative return as 1.0 (no change)
        double cumulative = 1.0;
        // Loop from just after entryIndex to and including exitIndex (inclusive trade interval)
        for (int i = entryIndex; i <= exitIndex; i++) {
            // For each bar/minute, multiply by the growth factor (1 + percent change/100)
            cumulative *= (1 + timeline.get(i).getPercentageChange() / 100);
        }
        // Apply cumulative compounded return to starting capital and return the result
        return capital * cumulative;
    }

    /**
     * Builds a lookup map for a given symbol, mapping each timestamp (LocalDateTime)
     * to its index in the price timeline. Used for fast retrieval.
     *
     * @param symbol   Stock symbol (e.g. "QBTS").
     * @param timeline List of StockUnit objects for the symbol.
     */
    private static void buildTimeIndex(String symbol, List<StockUnit> timeline) {
        // Prepare a fresh index map: LocalDateTime -> index
        Map<LocalDateTime, Integer> indexMap = new HashMap<>();
        // For every price bar in the timeline...
        for (int i = 0; i < timeline.size(); i++) {
            // Map each bar's exact timestamp to its index position (assumes timestamps are unique per bar)
            indexMap.put(timeline.get(i).getLocalDateTimeDate(), i);
        }
        // Save the index map to the global lookup for this symbol (used by getIndexForTime)
        symbolTimeIndex.put(symbol, indexMap);
    }

    /**
     * Looks up the index in a symbol's timeline for a specific timestamp.
     *
     * @param symbol Stock symbol.
     * @param time   The LocalDateTime to look for.
     * @return Index of that time in the timeline, or null if not found.
     */
    private static Integer getIndexForTime(String symbol, LocalDateTime time) {
        // Query the global index map for this symbol; if missing, use an empty map (prevents NPE)
        // Return the index for this exact timestamp, or null if absent
        return symbolTimeIndex.getOrDefault(symbol, Collections.emptyMap()).get(time);
    }

    /**
     * Logs final trading simulation results, including revenue, number of trades,
     * and average revenue per trade. Profits are taxed at 25%, losses are not.
     *
     * @param capital Ending capital after all trades.
     * @param initial Starting capital.
     * @param calls   Number of successful trades made.
     */
    private static void logFinalResults(double capital, double initial, int calls) {
        double revenue;

        // If profited, apply a 25% capital gains tax; if loss, no tax (simulate real-world trading tax regime)
        if ((capital - initial) > 0) {
            revenue = (capital - initial) * 0.75;
        } else {
            revenue = (capital - initial);
        }

        // Print summary results if there were any trades entered
        if (calls > 0) {
            System.out.printf("Total Revenue: €%.2f%n", revenue);
            System.out.printf("Successful Calls: %d%n", calls);
            System.out.printf("Revenue/Call: €%.2f%n%n", revenue / calls);
        }
    }

    /**
     * Loads and processes data for each symbol. Ensures directory exists, reads stock data,
     * and computes percentage changes for every StockUnit. Also triggers calculation of indicators
     * and labeling of price spikes.
     */
    private static void prepData() {
        // --- FILESYSTEM PREP ---
        File cacheDir = new File(CACHE_DIR);
        if (!cacheDir.exists()) cacheDir.mkdirs(); // Ensure cache directory exists for caching price files

        // --- PER-SYMBOL DATA LOAD AND PARSE ---
        Arrays.stream(pLTester.SYMBOLS).forEach(symbol -> {
            try {
                // Compose file name and path for this symbol's data file in the cache directory
                String fileName = symbol + ".txt";
                String cachePath = Paths.get(CACHE_DIR, fileName).toString();
                // Process and parse data from file (populates symbolTimelines, handles price objects)
                processStockDataFromFile(cachePath, symbol, pLTester.cut);
            } catch (IOException e) {
                e.printStackTrace();
            }
        });

        // --- PERCENTAGE CHANGE CALCULATION ---
        // For every loaded timeline...
        synchronized (symbolTimelines) {
            symbolTimelines.forEach((symbol, timeline) -> {
                // Skip if not enough data (needs at least 2 bars for % change calc)
                if (timeline.size() < 2) {
                    logTextArea.append("Not enough data for " + symbol + "\n");
                    return;
                }

                // Calculate percentage change (close-to-close) for each bar (relative to previous bar)
                for (int i = 1; i < timeline.size(); i++) {
                    StockUnit current = timeline.get(i);
                    StockUnit previous = timeline.get(i - 1);

                    if (previous.getClose() > 0) {
                        double change = ((current.getClose() - previous.getClose()) / previous.getClose()) * 100;
                        // Abnormal outliers (>= 14%) are set to the previous % change (keeps continuity for flash spikes)
                        change = Math.abs(change) >= 14 ? previous.getPercentageChange() : change;
                        current.setPercentageChange(change);
                    }
                }
            });
        }

        // --- INDICATOR AND LABELING PIPELINE ---
        // Precompute all indicator min/max ranges for normalization (not using live data)
        precomputeIndicatorRanges(false);

        // Recalculate percentage change labels and spike detection (for training, backtest, etc.)
        dataTester.calculateStockPercentageChange();

        // Compute and label "spikes" (large moves) in all timelines for later analysis
        calculateSpikesInRally(frameSize, false);
    }

    /**
     * Reads stock data from a given file, parses into StockUnit objects,
     * and appends them to the main timeline for the symbol. Targets (labels)
     * are reset to zero by default.
     *
     * @param filePath   Path to the cached file (e.g. QBTS.txt)
     * @param symbol     The stock symbol (e.g. QBTS)
     * @param retainLast Number of recent data bars to retain (for memory/performance)
     * @throws IOException If file is missing or unreadable
     */
    public static void processStockDataFromFile(String filePath, String symbol, int retainLast) throws IOException {
        // Parse file into StockUnit list
        List<StockUnit> fileUnits = readStockUnitsFromFile(filePath, retainLast);
        symbol = symbol.toUpperCase();

        // Set all targets to 0 (unlabeled by default)
        fileUnits.forEach(unit -> unit.setTarget(0));

        // Merge new data into main timeline for this symbol
        List<StockUnit> existing = symbolTimelines.getOrDefault(symbol, new ArrayList<>());
        existing.addAll(fileUnits);
        symbolTimelines.put(symbol, existing);
    }

    /**
     * Exports a list of StockUnits to a CSV file for ML model training or analysis.
     * If the file doesn't exist, a header is written. Appends new rows for each unit.
     *
     * @param stocks List of StockUnit objects to write
     */
    public static void exportToCSV(List<StockUnit> stocks) {
        try {
            Path filePath = Paths.get(System.getProperty("user.dir"), "rallyMLModel", "highFrequencyStocks.csv");
            File file = filePath.toFile();

            // Check if file exists to determine if headers should be added
            boolean fileExists = file.exists();

            // Open in append mode
            FileWriter csvWriter = new FileWriter(file, true);

            // Write CSV headers if new file
            if (!fileExists) {
                csvWriter.append("timestamp,open,high,low,close,volume,target\n");
            }

            // Write each StockUnit as a CSV row
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

    /**
     * Escapes a string for CSV output. If the string contains a comma, quote, or newline,
     * it will be enclosed in double quotes and all existing double quotes will be escaped.
     * Returns an empty string for null values.
     *
     * @param data The string to escape for CSV.
     * @return The escaped string, safe for CSV insertion.
     */
    private static String escapeCSV(String data) {
        if (data == null) {
            return "";  // Handle null values by returning an empty string
        }

        String escapedData = data; // Initialize escapedData with the original data

        // If data contains commas, quotes, or newlines, enclose it in double quotes and escape existing quotes
        if (data.contains(",") || data.contains("\"") || data.contains("\n")) {
            // Replace all " with "" and then surround the result with "
            escapedData = "\"" + data.replace("\"", "\"\"") + "\"";
        }
        return escapedData;
    }

    /**
     * Reads a list of StockUnit objects from a file, using a custom parsing strategy.
     * <ul>
     *     <li>Removes leading/trailing square brackets.</li>
     *     <li>Splits by '},' (end of a stock unit) to extract each record.</li>
     *     <li>Handles malformed lines and ensures robust parsing.</li>
     *     <li>Reverses order (for time ordering) and truncates to the last retainLast records.</li>
     * </ul>
     *
     * @param filePath   Path to the file to read (should be plain text, each StockUnit as string).
     * @param retainLast How many most recent entries to keep (for memory/performance).
     * @return List of StockUnit objects parsed from the file.
     * @throws IOException if file cannot be read.
     */
    public static List<StockUnit> readStockUnitsFromFile(String filePath, int retainLast) throws IOException {
        List<StockUnit> stockUnits = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            // Read the whole file as a single string and remove leading/trailing brackets
            String content = reader.lines()
                    .collect(Collectors.joining("\n"))
                    .replaceAll("^\\[|]$", "") // Remove array brackets if present
                    .trim();

            if (content.isEmpty()) {
                System.out.println("The file is empty or incorrectly formatted.");
                return stockUnits;
            }

            // Split by each closing brace that ends a StockUnit (note: this may need improvement for deeply nested structures)
            String[] entries = content.split("},\\s*");

            for (String entry : entries) {
                try {
                    entry = entry.trim();

                    // Remove final closing brace (if present)
                    if (entry.endsWith("}")) {
                        entry = entry.substring(0, entry.length() - 1);
                    }

                    // Remove StockUnit{ prefix for each entry, if present
                    entry = entry.replace("StockUnit{", "").trim();

                    // Now, try to parse the entry into a StockUnit object using your custom parser
                    stockUnits.add(parseStockUnit(entry));
                } catch (Exception e) {
                    // Print error but continue parsing next entries
                    e.printStackTrace();
                }
            }
        }

        // Reverse order to maintain most recent data at the end of the list
        Collections.reverse(stockUnits);

        // Only keep the last 'retainLast' entries
        int keepFrom;
        try {
            keepFrom = Math.max(0, stockUnits.size() - retainLast);
        } catch (Exception e) {
            keepFrom = 0;
        }
        return new ArrayList<>(stockUnits.subList(keepFrom, stockUnits.size()));
    }

    /**
     * Safely creates a GUI notification popup or log entry for a given Notification event.
     * Uses a wrapper to catch and print any errors encountered while displaying.
     *
     * @param currentEvent The notification to display (e.g. on chart, as popup, etc.)
     */
    private static void createNotification(Notification currentEvent) {
        try {
            // Calls a utility method to display a notification popup in the GUI.
            // Passes all relevant event data from the Notification object:
            // - Title and content for the notification window/message
            // - StockUnit list (the data points associated with this event)
            // - LocalDateTime of the event
            // - Stock symbol (for context)
            // - Percentage change or other value
            // - Config integer (used for color-coding or categorization)
            addNotification(
                    currentEvent.getTitle(),
                    currentEvent.getContent(),
                    currentEvent.getStockUnitList(),
                    currentEvent.getLocalDateTime(),
                    currentEvent.getSymbol(),
                    currentEvent.getChange(),
                    currentEvent.getConfig()
            );
        } catch (Exception e) {
            // If anything goes wrong (e.g. GUI error), print a stack trace for debugging
            e.printStackTrace();
        }
    }

    /**
     * Displays an interactive price chart for the given symbol using JFreeChart.
     * Supports both candlestick and time series line modes.
     * <ul>
     *     <li>Builds a dataset from StockUnit price data.</li>
     *     <li>Configures the renderer and axes for clear visuals.</li>
     *     <li>Adds notification markers and custom overlays.</li>
     *     <li>Creates a JFrame window for user interaction and labeling.</li>
     * </ul>
     *
     * @param symbol The stock symbol to chart (case-insensitive, ".txt" removed if present)
     */
    private static void createSuperChart(String symbol) {
        try {
            // Standardize the symbol string: uppercase and remove file extension (e.g. ".txt")
            String processedSymbol = symbol.toUpperCase().replace(".TXT", "");
            // Retrieve the complete timeline (list of StockUnit objects) for this symbol
            List<StockUnit> timeline = getSymbolTimeline(processedSymbol);

            // If no data is available for this symbol, print a warning and exit early
            if (timeline.isEmpty()) {
                System.out.println("No data available for " + processedSymbol);
                return;
            }

            JFreeChart chart;
            XYPlot plot;

            // ===== CANDLESTICK CHART MODE =====
            if (pLTester.useCandles) {
                // Create an array of OHLCDataItem for every StockUnit in the timeline
                OHLCDataItem[] dataItems = new OHLCDataItem[timeline.size()];
                for (int i = 0; i < timeline.size(); i++) {
                    StockUnit unit = timeline.get(i);
                    dataItems[i] = new OHLCDataItem(
                            unit.getDateDate(),   // Time/date of this candle/bar
                            unit.getOpen(),       // Open price for the period
                            unit.getHigh(),       // Highest price
                            unit.getLow(),        // Lowest price
                            unit.getClose(),      // Closing price
                            unit.getVolume()      // Trade volume
                    );
                }

                // Wrap the array into a DefaultOHLCDataset for JFreeChart
                OHLCDataset ohlcDataset = new DefaultOHLCDataset(
                        processedSymbol + " Candles",
                        dataItems
                );

                // Use JFreeChart's candlestick chart factory method
                chart = ChartFactory.createCandlestickChart(
                        processedSymbol + " Candlestick Chart",
                        "Time",    // X axis label
                        "Price",   // Y axis label
                        ohlcDataset,
                        true       // Show legend
                );

                // Retrieve the plot for further customization (background, axes, renderer, etc.)
                plot = chart.getXYPlot();

                // Auto-range the domain (X) axis
                plot.getDomainAxis().setAutoRange(true);
                // Configure the range (Y) axis: auto-range, but don't force zero into view (prices never go to 0)
                NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
                rangeAxis.setAutoRange(true);
                rangeAxis.setAutoRangeIncludesZero(false);
                rangeAxis.configure();

                // Configure the candlestick renderer for visual clarity
                CandlestickRenderer renderer = getCandlestickRenderer();
                plot.setRenderer(renderer);

                // ===== LINE CHART MODE =====
            } else {
                // For line chart, create separate time series for each price attribute
                TimeSeries closeSeries = new TimeSeries(processedSymbol + " Close");
                TimeSeries openSeries = new TimeSeries(processedSymbol + " Open");
                TimeSeries highSeries = new TimeSeries(processedSymbol + " High");
                TimeSeries lowSeries = new TimeSeries(processedSymbol + " Low");

                // Add each StockUnit to all four series with its corresponding price and timestamp
                for (StockUnit unit : timeline) {
                    closeSeries.add(new Second(unit.getDateDate()), unit.getClose());
                    openSeries.add(new Second(unit.getDateDate()), unit.getOpen());
                    lowSeries.add(new Second(unit.getDateDate()), unit.getLow());
                    highSeries.add(new Second(unit.getDateDate()), unit.getHigh());
                }

                // Combine the four series into a dataset for JFreeChart
                TimeSeriesCollection priceDataset = new TimeSeriesCollection();
                priceDataset.addSeries(closeSeries); // Index 0: Close (black)
                priceDataset.addSeries(highSeries);  // Index 1: High (red)
                priceDataset.addSeries(lowSeries);   // Index 2: Low (blue)
                priceDataset.addSeries(openSeries);  // Index 3: Open (green)

                // Build the line chart (time series chart) using JFreeChart's factory
                chart = ChartFactory.createTimeSeriesChart(
                        processedSymbol + " Analysis",
                        "Time",     // X axis label
                        "Price",    // Y axis label
                        priceDataset,
                        true,       // Show legend
                        true,       // Enable tooltips
                        false       // No URL generation
                );

                plot = chart.getXYPlot();

                // Configure renderer styles for each series: color, thickness, no points/markers
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

            // ===== COMMON PLOT STYLING =====
            chart.setBackgroundPaint(Color.WHITE);         // White background for the chart
            plot.setBackgroundPaint(Color.WHITE);          // White background for the plot area
            plot.setDomainGridlinePaint(Color.LIGHT_GRAY); // Light gray gridlines for clarity
            plot.setRangeGridlinePaint(Color.LIGHT_GRAY);

            // Add vertical event markers for key notifications (see method for logic)
            addNotificationMarkers(plot, processedSymbol, timeline);

            // ===== GUI ASSEMBLY: ADD CHART TO A FRAME WITH CONTROLS =====
            // Create the custom interactive ChartPanel with overlays, drag labeling, etc.
            ChartPanel chartPanel = createChartPanel(chart, processedSymbol);
            // Get the control panel for saving, clearing, and showing % change
            JPanel controlPanel = getControlPanel(chartPanel, processedSymbol);

            // Set up the JFrame for viewing the chart and interacting with controls
            JFrame frame = new JFrame("Price Timeline Analysis");
            frame.setLayout(new BorderLayout());
            frame.add(chartPanel, BorderLayout.CENTER);  // Main chart area
            frame.add(controlPanel, BorderLayout.SOUTH); // Controls below chart
            frame.pack();                                // Size frame to fit content
            frame.setSize(1700, 1000);                   // Large window for data
            frame.setLocation(60, 20);                   // Place near top left of screen
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // Exit app on close
            frame.setVisible(true);

        } catch (Exception e) {
            // Print error for debugging if anything goes wrong in chart setup
            e.printStackTrace();
        }
    }

    @NotNull
    private static CandlestickRenderer getCandlestickRenderer() {
        CandlestickRenderer renderer = new CandlestickRenderer();
        renderer.setAutoWidthMethod(CandlestickRenderer.WIDTHMETHOD_SMALLEST); // Narrow candles for high-frequency data
        renderer.setUpPaint(Color.GREEN);    // Candle color for up (close >= open)
        renderer.setDownPaint(Color.RED);    // Candle color for down (close < open)
        renderer.setUseOutlinePaint(true);   // Draw black border around candles
        renderer.setDrawVolume(false);
        return renderer;
    }

    /**
     * Creates a ChartPanel for a given JFreeChart, adding interactive event listeners
     * for labeling and measurement:
     * <ul>
     *   <li>Allows users to click-drag on the chart to label time intervals (for ML or backtesting).</li>
     *   <li>When shift is held, native zoom and pan is enabled; otherwise, drag is used for labeling.</li>
     *   <li>Adds support for crosshair cursor, keyboard focus, and dynamic overlays.</li>
     *   <li>Supports real-time preview of selected intervals.</li>
     * </ul>
     *
     * @param chart  The JFreeChart object to display.
     * @param symbol The stock symbol being displayed, used for updating interval labels.
     * @return Configured ChartPanel with all event listeners.
     */
    private static ChartPanel createChartPanel(JFreeChart chart, String symbol) {
        // Array to hold the starting point of a drag for interval labeling (null when not dragging)
        final Point2D[] dragStartPoint = {null};
        // Boolean array to track if zoom mode is active (true when shift is held down)
        final boolean[] isZoomMode = {false};

        // Create a custom ChartPanel with extra painting logic for labeled intervals/overlays
        ChartPanel chartPanel = getStockPanel(chart, dragStartPoint);

        // Attach the marker logic for measuring percentage change between two points
        createPercentageMarkers(chart, chartPanel);

        // By default, enable built-in zoom and scroll features for the chart
        chartPanel.setDomainZoomable(true);
        chartPanel.setRangeZoomable(true);
        chartPanel.setMouseWheelEnabled(true);

        // --- MOUSE LISTENER FOR LABELING (Click-drag to select intervals) ---
        chartPanel.addMouseListener(new MouseAdapter() {
            private double dragStartTime; // Stores the chart domain value (e.g. timestamp) where drag started

            @Override
            public void mousePressed(MouseEvent e) {
                // If shift is held, enter zoom mode (native JFreeChart behavior)
                isZoomMode[0] = e.isShiftDown();

                if (!isZoomMode[0]) {
                    // If not zooming, prepare for custom interval labeling
                    // Disable native zoom so we can handle drag logic ourselves
                    chartPanel.setDomainZoomable(false);
                    chartPanel.setRangeZoomable(false);
                    chartPanel.setMouseWheelEnabled(false);

                    // Record where the drag started, in Java2D chart pixel space
                    dragStartPoint[0] = chartPanel.translateScreenToJava2D(e.getPoint());
                    // Convert drag start X (in pixels) to domain axis value (e.g. timestamp in ms)
                    dragStartTime = chart.getXYPlot().getDomainAxis().java2DToValue(
                            dragStartPoint[0].getX(),
                            chartPanel.getScreenDataArea(),
                            chart.getXYPlot().getDomainAxisEdge()
                    );
                    // Prevent other listeners (e.g. JFreeChart native zoom) from acting
                    e.consume();
                }
            }

            @Override
            public void mouseReleased(MouseEvent e) {
                // If not in zoom mode and a drag actually occurred
                if (!isZoomMode[0] && dragStartPoint[0] != null) {
                    // Get where the drag ended, in Java2D pixel coordinates
                    Point2D endPoint = chartPanel.translateScreenToJava2D(e.getPoint());
                    XYPlot plot = chart.getXYPlot();
                    // Convert end X (in pixels) to domain axis value (e.g. timestamp)
                    double endTime = plot.getDomainAxis().java2DToValue(
                            endPoint.getX(),
                            chartPanel.getScreenDataArea(),
                            plot.getDomainAxisEdge()
                    );

                    // Add a new interval record from start to end (always left-to-right)
                    labeledIntervals.add(new TimeInterval(
                            Math.min(dragStartTime, endTime),
                            Math.max(dragStartTime, endTime)
                    ));

                    // For all StockUnits in the interval, set their target label to 1 (selected)
                    updateLabelsForInterval(symbol, dragStartTime, endTime, 1);

                    // Redraw overlays to show the new label region
                    chartPanel.repaint();
                    e.consume();
                }

                // Restore built-in zoom, pan, and mouse wheel when finished
                chartPanel.setDomainZoomable(true);
                chartPanel.setRangeZoomable(true);
                chartPanel.setMouseWheelEnabled(true);

                // Reset drag state for next operation
                dragStartPoint[0] = null;
                isZoomMode[0] = false;
            }
        });

        // --- MOUSE MOTION LISTENER FOR REAL-TIME FEEDBACK DURING LABELING DRAG ---
        chartPanel.addMouseMotionListener(new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                if (!isZoomMode[0]) {
                    // Only repaint overlays if we're actively labeling (not zooming)
                    chartPanel.repaint();
                    e.consume();
                }
            }
        });

        // --- KEY LISTENER FOR SHIFT KEY (CURSOR CHANGES FOR FEEDBACK) ---
        chartPanel.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                // Change cursor to crosshair if shift is pressed (signals zoom mode)
                if (e.getKeyCode() == KeyEvent.VK_SHIFT) {
                    chartPanel.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
                }
            }

            @Override
            public void keyReleased(KeyEvent e) {
                // Restore cursor to default when shift is released
                if (e.getKeyCode() == KeyEvent.VK_SHIFT) {
                    chartPanel.setCursor(Cursor.getDefaultCursor());
                }
            }
        });

        // Make sure chartPanel can receive keyboard events (needed for shift/zoom detection)
        chartPanel.setFocusable(true);
        chartPanel.requestFocusInWindow();

        // Return the fully interactive panel for display in your JFrame
        return chartPanel;
    }

    /**
     * Returns a ChartPanel with a custom paintComponent implementation that draws:
     * <ul>
     *     <li>Previously labeled intervals as translucent green overlays</li>
     *     <li>The currently dragged interval (blue overlay) if mouse is held down</li>
     * </ul>
     * This allows real-time feedback for users labeling the chart.
     *
     * @param chart          The JFreeChart to display.
     * @param dragStartPoint Reference to the drag start position (null if not dragging).
     * @return A custom ChartPanel with overlay rendering logic.
     */
    @NotNull
    private static ChartPanel getStockPanel(JFreeChart chart, Point2D[] dragStartPoint) {
        // Return a custom ChartPanel that overrides paintComponent to draw extra overlays for labeling
        return new ChartPanel(chart) {
            @Override
            public void paintComponent(Graphics g) {
                // Always call super first to draw the regular chart background, grid, and series
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g.create();

                // Obtain the plot and axis for data-to-pixel conversion
                XYPlot plot = chart.getXYPlot();
                Rectangle2D dataArea = this.getScreenDataArea(); // Area where the chart is actually drawn
                ValueAxis domainAxis = plot.getDomainAxis();

                // --- Draw previously labeled intervals as translucent green overlays ---
                for (TimeInterval interval : labeledIntervals) {
                    // Convert interval start and end (in domain units, e.g., ms since epoch) to pixel X coordinates
                    double startX = domainAxis.valueToJava2D(interval.startTime, dataArea, plot.getDomainAxisEdge());
                    double endX = domainAxis.valueToJava2D(interval.endTime, dataArea, plot.getDomainAxisEdge());

                    // Set overlay color: transparent green (alpha 30/255)
                    g2.setColor(new Color(0, 255, 0, 30));
                    // Draw a filled rectangle across the entire chart height between startX and endX
                    g2.fillRect(
                            (int) Math.min(startX, endX), // Left X (min)
                            0,                            // Top of the chart panel
                            (int) Math.abs(endX - startX),// Width = difference between endpoints
                            this.getHeight()              // Full vertical span of chart
                    );
                }

                // --- Draw current drag preview as a translucent blue overlay ---
                if (dragStartPoint[0] != null) {
                    // If the user is currently dragging, get their current mouse position in Java2D coordinates
                    Point2D currentPoint = this.getMousePosition();
                    if (currentPoint != null) {
                        // Convert mouse screen coordinates to plot Java2D coordinates
                        currentPoint = translateScreenToJava2D((Point) currentPoint);

                        // Convert drag start and current X to domain value, then back to Java2D X (pixels)
                        double startX = domainAxis.valueToJava2D(
                                domainAxis.java2DToValue(dragStartPoint[0].getX(), dataArea, plot.getDomainAxisEdge()),
                                dataArea, plot.getDomainAxisEdge()
                        );
                        double currentX = domainAxis.valueToJava2D(
                                domainAxis.java2DToValue(currentPoint.getX(), dataArea, plot.getDomainAxisEdge()),
                                dataArea, plot.getDomainAxisEdge()
                        );

                        // Set overlay color: transparent blue (alpha 30/255)
                        g2.setColor(new Color(0, 0, 255, 30));
                        // Draw a filled rectangle between drag start and current X, covering the full chart height
                        g2.fillRect(
                                (int) Math.min(startX, currentX), // Left edge of rectangle
                                0,                                // Top
                                (int) Math.abs(currentX - startX),// Width
                                this.getHeight()                  // Full height
                        );
                    }
                }

                // Clean up resources
                g2.dispose();
            }
        };
    }

    /**
     * Adds a mouse listener to the ChartPanel so that the user can click two points
     * on the chart to mark an interval for measuring percentage change.
     * <ul>
     *   <li>First click: places the starting marker (green vertical line).</li>
     *   <li>Second click: places the ending marker, shades the region, and displays
     *   the percentage change between the two prices.</li>
     * </ul>
     *
     * @param chart      The JFreeChart object (for retrieving axes and plot).
     * @param chartPanel The ChartPanel on which mouse events will be handled.
     */
    private static void createPercentageMarkers(JFreeChart chart, ChartPanel chartPanel) {
        chartPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                // Convert mouse screen coordinates to Java2D coordinates (plot pixel space)
                Point2D p = chartPanel.translateScreenToJava2D(e.getPoint());
                XYPlot plot = chart.getXYPlot();
                ValueAxis xAxis = plot.getDomainAxis();
                ValueAxis yAxis = plot.getRangeAxis();

                // Convert pixel position to actual data values (x = time, y = price)
                double x = xAxis.java2DToValue(p.getX(), chartPanel.getScreenDataArea(), plot.getDomainAxisEdge());
                double y = yAxis.java2DToValue(p.getY(), chartPanel.getScreenDataArea(), plot.getRangeAxisEdge());

                // If first point not set, set it and draw marker
                if (Double.isNaN(point1X)) {
                    point1X = x;
                    point1Y = y;
                    addFirstMarker(plot, point1X);
                } else {
                    // Otherwise, set the second point, calculate change, draw everything, and reset
                    point2X = x;
                    point2Y = y;
                    addSecondMarkerAndShade(plot);
                    resetPoints();
                }
            }
        });
    }

    /**
     * Builds and returns a JPanel containing buttons and a percentage display for chart controls.
     * Buttons allow user to:
     * <ul>
     *   <li>Save all labels to CSV</li>
     *   <li>Clear the last labeled interval</li>
     *   <li>Restore chart axis to auto range</li>
     * </ul>
     * Also includes a JLabel that displays the most recent percentage change computed.
     *
     * @param chartPanel The ChartPanel being controlled.
     * @param symbol     The stock symbol being analyzed (used for saving/clearing labels).
     * @return JPanel with all controls.
     */
    private static JPanel getControlPanel(ChartPanel chartPanel, String symbol) {
        // Create the "Save All Labels" button and attach an action to save all interval labels to CSV
        JButton saveButton = new JButton("Save All Labels");
        saveButton.addActionListener(e -> saveLabels(symbol)); // Save all labeled data to CSV

        // Create the "Clear Last" button and attach an action to remove the most recent labeled interval
        JButton clearButton = new JButton("Clear Last");
        clearButton.addActionListener(e -> clearLastInterval(chartPanel, symbol)); // Remove last label

        // Create the "Auto Range" button to reset the chart's zoom to default bounds
        JButton autoRangeButton = new JButton("Auto Range");
        autoRangeButton.addActionListener(e -> chartPanel.restoreAutoBounds()); // Reset zoom

        // Label to show the current percentage change between two markers (updated dynamically)
        percentageChange = new JLabel("Percentage Change");

        // Assemble all controls into a panel (horizontal layout by default)
        JPanel panel = new JPanel();
        panel.add(autoRangeButton);   // Add zoom reset button
        panel.add(saveButton);        // Add save labels button
        panel.add(clearButton);       // Add clear last label button
        panel.add(percentageChange);  // Add dynamic percentage label

        return panel; // Return the fully configured panel
    }

    /**
     * Saves all labels and timeline data for the given symbol to a CSV file.
     * Uses the global symbolTimelines map.
     * Pops up a dialog to notify the user when done.
     *
     * @param symbol The stock symbol being saved.
     */
    private static void saveLabels(String symbol) {
        // Get the timeline (list of StockUnit objects) for the given symbol
        List<StockUnit> timeline = symbolTimelines.get(symbol);

        // If the timeline exists (symbol found)
        if (timeline != null) {
            // Export all data (with current target labels) to a CSV file
            exportToCSV(timeline);

            // Show a popup message to the user confirming the save action
            JOptionPane.showMessageDialog(null, "All labels saved successfully!");
        }
    }

    /**
     * Removes the most recently labeled interval and updates the StockUnit labels
     * to clear its target flag. Also repaints the chart to reflect the removal.
     *
     * @param chartPanel The ChartPanel to repaint after clearing.
     * @param symbol     The symbol whose timeline to update.
     */
    private static void clearLastInterval(ChartPanel chartPanel, String symbol) {
        if (labeledIntervals.isEmpty()) return;

        // Remove the last interval
        TimeInterval last = labeledIntervals.remove(labeledIntervals.size() - 1);
        // Set target back to 0 for all units in that interval
        updateLabelsForInterval(symbol, last.startTime, last.endTime, 0);
        chartPanel.repaint();
    }

    /**
     * Updates the target label for all StockUnits in a timeline that fall within a specified interval.
     * Used for both labeling (target=1) and clearing (target=0) intervals.
     *
     * @param symbol    The stock symbol to update.
     * @param startTime The lower bound of the interval (x-axis value, ms since epoch).
     * @param endTime   The upper bound of the interval.
     * @param value     The label to set (1 = labeled, 0 = not labeled).
     */
    private static void updateLabelsForInterval(String symbol, double startTime, double endTime, int value) {
        // Retrieve the timeline (list of StockUnit) for the given symbol
        List<StockUnit> timeline = symbolTimelines.get(symbol);
        if (timeline == null) return; // If symbol not found, do nothing

        // Iterate over every StockUnit in the timeline
        timeline.stream()
                .filter(unit -> {
                    // Convert the StockUnit's timestamp (Date) to milliseconds
                    long unitTime = unit.getDateDate().getTime();
                    // Only include units whose timestamp is within [startTime, endTime] (inclusive)
                    return unitTime >= startTime && unitTime <= endTime;
                })
                .forEach(unit -> unit.setTarget(value)); // Set the target (label) to the specified value
    }

    /**
     * Adds vertical markers to the chart at the times of all notifications for a given symbol.
     * Marker color depends on the notification config field.
     *
     * @param plot     The XYPlot to add markers to.
     * @param symbol   The symbol currently being plotted.
     * @param timeline The list of StockUnit objects for the symbol.
     */
    private static void addNotificationMarkers(XYPlot plot, String symbol, List<StockUnit> timeline) {
        // Loop through all notifications to add markers for relevant events
        for (Notification notification : notificationsForPLAnalysis) {
            // Only process notifications matching the given symbol
            if (!notification.getSymbol().equalsIgnoreCase(symbol)) continue;

            // Get the event's timestamp
            LocalDateTime notifyTime = notification.getLocalDateTime();
            // Find the index in the timeline for this notification
            Integer index = getIndexForTime(symbol, notifyTime);

            // If a valid index and StockUnit exist, add a vertical marker to the plot
            if (index != null && index < timeline.size()) {
                StockUnit unit = timeline.get(index);
                ValueMarker marker = getValueMarker(notification, unit);
                plot.addDomainMarker(marker); // Draws a vertical colored line at the event time
            }
        }
    }

    /**
     * Creates and returns a ValueMarker (vertical line) for a notification, with color
     * and style based on its configuration.
     *
     * @param notification The Notification object.
     * @param unit         The StockUnit associated with the notification's time.
     * @return The ValueMarker to be added to the plot.
     */
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

    /**
     * Adds the first percentage marker (vertical green line) at the specified x-position,
     * clearing any existing markers or shaded regions before adding.
     *
     * @param plot      The XYPlot to add the marker to.
     * @param xPosition The domain axis value (timestamp in ms) for the marker.
     */
    private static void addFirstMarker(XYPlot plot, double xPosition) {
        // Remove any previous markers or shaded regions
        if (marker1 != null) {
            plot.removeDomainMarker(marker1);
        }
        if (marker2 != null) {
            plot.removeDomainMarker(marker2);
        }
        if (shadedRegion != null) {
            plot.removeDomainMarker(shadedRegion);
        }

        // Create and add the new marker
        marker1 = new ValueMarker(xPosition);
        marker1.setPaint(Color.GREEN);  // First marker is green
        marker1.setStroke(new BasicStroke(1.5f));  // Slightly thicker line
        plot.addDomainMarker(marker1);
    }

    /**
     * Adds the second marker (color depends on gain/loss), shades the region between
     * markers, and updates the displayed percentage change.
     * Clears any previous second markers or shaded regions first.
     *
     * @param plot The XYPlot to update.
     */
    private static void addSecondMarkerAndShade(XYPlot plot) {
        // Remove any previous second marker or shaded region
        if (marker2 != null) {
            plot.removeDomainMarker(marker2);
        }
        if (shadedRegion != null) {
            plot.removeDomainMarker(shadedRegion);
        }

        // Calculate percentage difference between y-values of markers
        double percentageDiff = ((point2Y - point1Y) / point1Y) * 100;

        // Color green for gain, red for loss
        Color markerColor = (percentageDiff >= 0) ? Color.GREEN : Color.RED;
        Color shadeColor = (percentageDiff >= 0) ? new Color(100, 200, 100, 50) : new Color(200, 100, 100, 50);

        // Add second vertical marker at the second clicked point
        marker2 = new ValueMarker(point2X);
        marker2.setPaint(markerColor);
        marker2.setStroke(new BasicStroke(1.5f));
        plot.addDomainMarker(marker2);

        // Add a translucent shaded region between the two markers
        shadedRegion = new IntervalMarker(Math.min(point1X, point2X), Math.max(point1X, point2X));
        shadedRegion.setPaint(shadeColor);
        plot.addDomainMarker(shadedRegion);

        // Update the percentage change label in the control panel
        percentageChange.setText(String.format("Percentage Change: %.3f%%", percentageDiff));
    }

    /**
     * Resets the marker points to NaN so a new interval can be started/measured.
     * Used after each complete measurement.
     */
    private static void resetPoints() {
        point1X = Double.NaN;
        point1Y = Double.NaN;
        point2X = Double.NaN;
        point2Y = Double.NaN;
    }

    /**
     * Record representing a labeled interval on the timeline.
     * Stores start and end domain axis values (usually timestamps).
     */
    private record TimeInterval(double startTime, double endTime) {
    }
}