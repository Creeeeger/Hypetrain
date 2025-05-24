package org.crecker;

import com.crazzyghost.alphavantage.news.response.NewsResponse;
import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import com.formdev.flatlaf.themes.FlatMacDarkLaf;
import org.jetbrains.annotations.NotNull;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.CandlestickRenderer;
import org.jfree.data.time.*;
import org.jfree.data.time.ohlc.OHLCItem;
import org.jfree.data.time.ohlc.OHLCSeries;
import org.jfree.data.time.ohlc.OHLCSeriesCollection;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.crecker.configHandler.createConfig;
import static org.crecker.mainDataHandler.CACHE_DIR;

/**
 * mainUI is the primary graphical user interface class for the "Hype train" stock monitoring application.
 * <p>
 * This JFrame subclass orchestrates all UI components, manages configuration loading, handles user interactions
 * with stocks, notifications, and news, and serves as the central coordinator between user actions and backend logic.
 * The class is designed as a singleton to ensure only one instance manages the application's window and its state.
 * <p>
 * Core responsibilities:
 * <ul>
 *   <li>Initializes the frame, menu bar, and all side panels (symbols, chart tools, notifications/hype panel).</li>
 *   <li>Handles loading and saving of user settings via configuration files.</li>
 *   <li>Provides static access and updates to UI elements reflecting real-time stock data, charts, and alerts.</li>
 *   <li>Integrates with mainDataHandler and other backend classes for asynchronous operations.</li>
 *   <li>Manages stock list, chart refresh logic, color-coding, and real-time UI updates for selected stocks.</li>
 * </ul>
 * <p>
 * This class uses various static fields to share state across the UI, reflecting design decisions to ensure single-source
 * synchronization for all major components such as the stock list, log window, and selected stock data.
 */
public class mainUI extends JFrame {
    /**
     * Trading212 authentication token (API key) & PushCut API token required for secure API requests.
     */
    static String t212ApiToken; // Trading212 API token
    static String pushCutUrlEndpoint; // PushCut url notification String

    /**
     * Mapping from company short names to TickerData (symbol, max position size). Used for search and symbol lookup.
     */
    public static final Map<String, TickerData> nameToData = new TreeMap<>();

    /**
     * Single-threaded executor used for periodic background tasks (e.g., real-time stock price refresh).
     */
    private static final ScheduledExecutorService executorService = Executors.newSingleThreadScheduledExecutor();

    /**
     * Caching structure for aggregated chart data, mapping stock symbols and time bucket info
     * to precalculated OHLC (Open, High, Low, Close) data for efficiency.
     */
    private static final Map<String, Map<RegularTimePeriod, AggregatedStockData>> aggregationCache = new ConcurrentHashMap<>();

    /**
     * Log area for system messages and user-visible notifications at the bottom of the hype panel.
     */
    public static JTextArea logTextArea;

    /**
     * The main application window (singleton instance).
     */
    public static mainUI gui;

    /**
     * Flag indicating whether the candlestick chart view is enabled (true for OHLC, false for line chart).
     */
    public static boolean useCandles;

    /**
     * Flag indicating whether the greed Mode is enabled.
     */
    public static boolean greed;

    /**
     * Current volume parameter for algorithms (user-configurable).
     */
    static int volume;

    /**
     * Current aggressiveness setting for hype mode or algorithmic trading.
     */
    static float aggressiveness;

    /**
     * Global flag: whether notifications should be sorted (affects hype panel).
     */
    static boolean shouldSort, useRealtime;

    /**
     * Global flag for whether notification sorting is by % change or date.
     */
    static boolean globalByChange = false;

    /**
     * String representation of tracked symbols in the watchlist, persisted in config.
     */
    static String symbols, apiKey;

    /**
     * The currently selected stock ticker symbol (for chart, info, and news panels).
     */
    static String selectedStock = "-Select a Stock-";

    // Selection of the market to hype in
    public static String market;
    /**
     * UI panels for layout: symbol list, chart/tool section, notifications/hype panel, and chart panel itself.
     */
    static JPanel symbolPanel, chartToolPanel, hypePanel, chartPanel;

    /**
     * Text field for symbol search input at the top of the symbol panel.
     */
    static JTextField searchField;

    // --- Chart navigation buttons (time range selectors) ---
    static JButton oneMinutesButton, threeMinutesButton, fiveMinutesButton, tenMinutesButton, thirtyMinutesButton,
            oneHourButton, fourHourButton, oneDayButton, threeDaysButton, oneWeekButton, twoWeeksButton, oneMonthButton;

    // --- Stock info display labels ---
    static JLabel openLabel, highLabel, lowLabel, volumeLabel, peLabel, mktCapLabel,
            fiftyTwoWkHighLabel, fiftyTwoWkLowLabel, pegLabel, percentageChange;

    /**
     * Model holding the current watchlist (user's chosen stocks, as strings).
     */
    static DefaultListModel<String> stockListModel;

    /**
     * Mapping of stock tickers to colors for UI representation in lists and charts.
     */
    static Map<String, Color> stockColors;

    /**
     * The main ChartPanel component, displays live price or OHLC chart.
     */
    static ChartPanel chartDisplay;

    /**
     * JFreeChart time series (line chart) and OHLCSeries (candlestick chart) datasets.
     */
    static TimeSeries timeSeries;
    static OHLCSeries ohlcSeries;

    /**
     * Placeholder panel to swap in and out chart components.
     */
    static JPanel chartPlaceholder;

    /**
     * Flag to keep track of whether current chart mode is candlestick (prevents redundant refresh).
     */
    static boolean modeCopy;

    /**
     * Code of the currently selected time range for chart (e.g., 1-min, 1-hour).
     */
    static int currentTimeRangeChoice = 9;

    // --- Notification UI state ---
    static JList<Notification> notificationList;
    static DefaultListModel<Notification> notificationListModel;
    static Notification currentNotification; // Track currently opened notification

    // --- News UI state ---
    static DefaultListModel<News> NewsListModel;
    static News CurrentNews;

    /**
     * The currently loaded series of stock data for the selected stock, in memory.
     */
    private static List<StockUnit> stocks = Collections.synchronizedList(new ArrayList<>());

    // --- Interactive marker state for chart annotation (used in percentage change tools) ---
    private static double point1X = Double.NaN;
    private static double point1Y = Double.NaN;
    private static double point2X = Double.NaN;
    private static double point2Y = Double.NaN;
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;

    /**
     * Singleton pattern: stores the currently active mainUI instance.
     */
    private static mainUI instance;

    /**
     * Tracks which stock symbol is currently shown in the chart (avoid unnecessary reloads).
     */
    private static String currentStockSymbol = null;

    /**
     * Remembers the last selected time window (for efficiency in chart refresh).
     */
    private static int lastChoice = -1;

    /**
     * Constructs the main user interface for the stock monitor.
     * Initializes all UI panels, sets up the menu bar, and assigns layout.
     * Also sets OS-level taskbar icon (where supported), and configures the panel hierarchy:
     * Symbol panel (left), Chart/tools (center), Hype/notifications (right).
     */
    public mainUI() {
        instance = this;

        // Configure layout: BorderLayout, with each major panel in a separate section
        setLayout(new BorderLayout());
        BorderFactory.createTitledBorder("Stock monitor");

        // Attempt to set a custom icon on supported operating systems
        try {
            if (Taskbar.isTaskbarSupported()) {
                final Taskbar taskbar = Taskbar.getTaskbar();
                taskbar.setIconImage(new ImageIcon(Paths.get(System.getProperty("user.dir"), "train.png").toString()).getImage());
            }
        } catch (Exception ignored) {
            // Windows may throw exceptions; ignore if not supported
        }

        // Set up the main menu bar (File, Settings, Hype Mode, Notifications, etc.)
        setJMenuBar(createMenuBar());

        // Instantiate and attach the three core panels
        symbolPanel = createSymbolPanel();
        chartToolPanel = createChartToolPanel();
        hypePanel = createHypePanel();

        // Place them in the correct sections of the window
        add(symbolPanel, BorderLayout.WEST);
        add(chartToolPanel, BorderLayout.CENTER);
        add(hypePanel, BorderLayout.EAST);
    }

    /**
     * @return The singleton instance of mainUI, representing the running UI window.
     */
    public static mainUI getInstance() {
        return instance;
    }

    /**
     * Loads application settings from the configuration file (config.xml), or creates a new config if missing.
     * Initializes key static fields (volume, symbols, sorting flag, API key, etc.) from the loaded settings.
     * Should be called at application startup and after any import/load action.
     */
    public static void setValues() {
        try {
            // Load settings as 2D array of key-value pairs
            String[][] settingData = configHandler.loadConfig();
            volume = Integer.parseInt(settingData[0][1]);
            symbols = settingData[1][1];
            shouldSort = Boolean.parseBoolean(settingData[2][1]);
            apiKey = settingData[3][1];
            useRealtime = Boolean.parseBoolean(settingData[4][1]);
            aggressiveness = Float.parseFloat(settingData[5][1]);
            useCandles = Boolean.parseBoolean(settingData[6][1]);
            t212ApiToken = settingData[7][1];
            pushCutUrlEndpoint = settingData[8][1];
            greed = Boolean.parseBoolean(settingData[9][1]);
            market = settingData[10][1];
        } catch (Exception e) {
            System.out.println("Config error - Create new config " + e.getMessage());
            createConfig();
        }
    }

    /**
     * Main entry point for the application. Initializes settings, ensures necessary directories/files exist,
     * and launches the user interface. Loads or creates configuration as needed.
     * Sets up the application window, initial log area, and starts the chart and data fetchers.
     *
     * @param args Command-line arguments (unused).
     * @throws Exception If fatal error occurs during setup.
     */
    public static void main(String[] args) throws Exception {
        try {
            // --- Set up the FlatLaf Mac Dark Look and Feel for modern UI styling ---
            FlatMacDarkLaf.setup();

            // --- Customise global UI appearance for a more rounded look ---
            UIManager.put("Component.arc", 20);         // General components
            UIManager.put("Button.arc", 20);            // JButton
            UIManager.put("TextComponent.arc", 20);     // JTextField, JTextArea, etc.
            UIManager.put("ProgressBar.arc", 20);       // JProgressBar
            UIManager.put("CheckBox.arc", 20);          // JCheckBox
            UIManager.put("RadioButton.arc", 20);       // JRadioButton
            UIManager.put("MenuBar.arc", 20);           // JMenuBar
            UIManager.put("PopupMenu.arc", 20);         // JPopupMenu
            UIManager.put("Panel.arc", 20);             // JPanel
            UIManager.put("ScrollBar.thumbArc", 20);    // JScrollBar thumbs
            UIManager.put("Slider.trackArc", 20);       // JSlider track
            UIManager.put("Slider.thumbArc", 20);       // JSlider thumb
            UIManager.put("TabbedPane.tabArc", 20);     // JTabbedPane tabs
            UIManager.put("Table.arc", 20);             // JTable
            UIManager.put("TableHeader.arc", 20);       // JTable header
            UIManager.put("ToolBar.arc", 20);           // JToolBar
            UIManager.put("ComboBox.arc", 20);          // JComboBox

        } catch (Exception ex) {
            // If Look and Feel setup fails, print error details for debugging
            ex.printStackTrace();
        }

        // Ensure the cache directory for downloaded data exists
        File cacheDir = new File(CACHE_DIR);
        if (!cacheDir.exists()) cacheDir.mkdirs();

        // Locate or create the main configuration file
        Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
        File config = configPath.toFile();
        if (!config.exists()) {
            // If no config exists, create one and show the settings window immediately
            createConfig();
            setValues();

            gui = new mainUI();
            gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            gui.setSize(1900, 1000); // Set window size (W x H)
            gui.setVisible(true);
            gui.setTitle("Hype train");

            // Set up initial labels for stock info section
            updateStockInfoLabels(0, 0, 0, 0, 0, 0, 0, 0, 0);

            // Load the initial stock symbol table (from config)
            loadTable(symbols);

            // Immediately open settings so user can edit config before using the app
            settingsHandler guiSetting = new settingsHandler(volume, symbols = createSymArray(), shouldSort, apiKey, useRealtime, aggressiveness,
                    useCandles, t212ApiToken, pushCutUrlEndpoint, greed, market);
            guiSetting.setSize(500, 700);
            guiSetting.setAlwaysOnTop(true);
            guiSetting.setModalityType(Dialog.ModalityType.APPLICATION_MODAL); // makes it blocking
            guiSetting.setLocationRelativeTo(null);
            guiSetting.setVisible(true);

            // Initialize the main API handler for stock price data
            if (!apiKey.isEmpty()) {
                mainDataHandler.InitAPi(apiKey);
            } else {
                throw new RuntimeException("You need to add a key in the settings menu first");
            }

            // Log new config event
            logTextArea.append("New config created\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        } else {
            // If config exists, load it and initialize the main UI as normal
            setValues();

            gui = new mainUI();
            gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            gui.setSize(1900, 1000);
            gui.setVisible(true);
            gui.setTitle("Hype train");

            updateStockInfoLabels(0, 0, 0, 0, 0, 0, 0, 0, 0);

            logTextArea.append("Load config\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

            if (!apiKey.isEmpty()) {
                mainDataHandler.InitAPi(apiKey);
            } else {
                throw new RuntimeException("You need to add a key in the settings menu first");
            }

            loadTable(symbols);
            logTextArea.append("Config loaded\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        }

        // Refresh the chart view with the latest settings (default to 3-day window)
        refreshChartType(false);
        // Fetch all available ticker metadata for searching and UI
        fetchTickerMap();
    }

    /**
     * Recursively refreshes all components within the specified container.
     * <p>
     * This method ensures all UI elements are revalidated and repainted, which is useful after
     * dynamic updates or when components need to reflect new data or configuration.
     * The call is wrapped with {@link SwingUtilities#invokeLater(Runnable)} to ensure
     * thread safety by updating on the Event Dispatch Thread (EDT).
     *
     * @param container The parent Swing Container whose child components are to be refreshed.
     */
    public static void refreshAllComponents(Container container) {
        SwingUtilities.invokeLater(() -> {
            for (Component component : container.getComponents()) {
                component.revalidate(); // Re-calculate layout for component
                component.repaint();    // Request visual refresh
                if (component instanceof Container) {
                    refreshAllComponents((Container) component);  // Recursively refresh nested containers
                }
            }
        });
    }

    /**
     * Saves the current application configuration using the supplied key-value pairs.
     * <p>
     * This wraps the save call and also logs the action in the logTextArea for user visibility.
     *
     * @param data 2D array of settings in the format { {key, value}, ... }
     */
    public static void saveConfig(String[][] data) {
        configHandler.saveConfig(data);
        logTextArea.append("Config saved successfully\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
    }

    /**
     * Loads the watchlist symbols and their associated colors from a configuration string.
     * <p>
     * The config string is expected in the format: [SYMBOL,java.awt.Color[r=R,g=G,b=B]],...
     * After parsing, updates the {@link #stockListModel} and {@link #stockColors} map.
     * <p>
     * If no previous elements are found (e.g. on fresh start), a log message is printed.
     *
     * @param config Serialized symbol and color configuration string.
     */
    public static void loadTable(String config) {
        // Clear previous colors and symbols
        stockColors.clear();
        stockListModel.clear();

        try {
            config = config.substring(1, config.length() - 1); // Remove outer brackets: "[...]" → "..."
            String[] entries = config.split("],\\["); // Split each symbol-color entry

            // Create a temporary 2D array for stock symbol and color objects
            Object[][] stockArray = new Object[entries.length][2];

            // Parse each entry for symbol and color
            for (int i = 0; i < entries.length; i++) {
                // Split by ",java.awt.Color[r=" to separate symbol and color
                String[] parts = entries[i].split(",java.awt.Color\\[r=");
                String stockSymbol = parts[0]; // e.g., "AAPL"
                String colorString = parts[1]; // e.g., "102,g=205,b=170]"

                // Parse the RGB values out of colorString
                String[] rgbParts = colorString.replace("]", "").split(",g=|,b=");
                int r = Integer.parseInt(rgbParts[0]);
                int g = Integer.parseInt(rgbParts[1]);
                int b = Integer.parseInt(rgbParts[2]);

                // Create a Color object from the RGB values
                Color color = new Color(r, g, b);

                // Add the Stock symbol and color to the 2D array
                stockArray[i][0] = stockSymbol;
                stockArray[i][1] = color;
            }

            // Add parsed values to the list model and color map
            for (Object[] objects : stockArray) {
                stockListModel.addElement(objects[0].toString());
                stockColors.put(objects[0].toString(), (Color) objects[1]);
            }
        } catch (Exception e) {
            // If parsing fails (e.g., empty config), log the error
            e.printStackTrace();
            logTextArea.append("No elements saved before\n");
            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
        }
    }

    /**
     * Serializes the current symbol watchlist and their color values into a config string.
     * <p>
     * Used for storing into the config.xml and exporting/importing settings.
     * The format is: [SYMBOL,java.awt.Color[r=...,g=...,b=...]],...
     *
     * @return The string representation of the symbol-color mapping for saving.
     */
    public static String createSymArray() {
        StringBuilder symBuilder = new StringBuilder();

        // Append each symbol and color as [symbol,color]
        for (Map.Entry<String, Color> entry : stockColors.entrySet()) {
            String stockSymbol = entry.getKey();
            Color color = entry.getValue();

            symBuilder.append("[")
                    .append(stockSymbol)
                    .append(",")
                    .append(color)
                    .append("],");
        }

        // Remove trailing comma if present
        if (!symBuilder.isEmpty()) {
            symBuilder.setLength(symBuilder.length() - 1);
        }
        return symBuilder.toString();
    }

    /**
     * Updates all the labels in the stock information panel with the latest market data.
     * <p>
     * All values are formatted for display. Used after selecting a stock or on periodic updates.
     *
     * @param open           Latest open price.
     * @param high           Latest high price.
     * @param low            Latest low price.
     * @param volume         Latest traded volume.
     * @param peRatio        Latest Price/Earnings ratio.
     * @param pegRatio       Latest P/E/Growth ratio.
     * @param fiftyTwoWkHigh Latest 52-week high.
     * @param fiftyTwoWkLow  Latest 52-week low.
     * @param marketCap      Latest market capitalization.
     */
    public static void updateStockInfoLabels(double open, double high, double low, double volume,
                                             double peRatio, double pegRatio, double fiftyTwoWkHigh, double fiftyTwoWkLow, double marketCap) {
        openLabel.setText("Open: " + String.format("%.2f", open));
        highLabel.setText("High: " + String.format("%.2f", high));
        lowLabel.setText("Low: " + String.format("%.2f", low));
        volumeLabel.setText("Vol: " + String.format("%.0f", volume));
        peLabel.setText("P/E: " + String.format("%.2f", peRatio));
        pegLabel.setText("P/E/G: " + String.format("%.0f", pegRatio));
        fiftyTwoWkHighLabel.setText("52W H: " + String.format("%.2f", fiftyTwoWkHigh));
        fiftyTwoWkLowLabel.setText("52W L: " + String.format("%.2f", fiftyTwoWkLow));
        mktCapLabel.setText("Mkt Cap: " + String.format("%.2f", marketCap));
    }

    /**
     * Assembles the current configuration into a 2D string array for saving.
     * <p>
     * Each entry is a {key, value} pair corresponding to a major setting, such as volume,
     * symbols, sort order, API key, real-time flag, algorithm aggressiveness, and chart type.
     *
     * @return The current configuration as a 2D string array.
     */
    public static String[][] getValues() {
        return new String[][]{
                {"volume", String.valueOf(volume)},
                {"symbols", symbols = createSymArray()},
                {"sort", String.valueOf(shouldSort)},
                {"key", apiKey},
                {"realtime", String.valueOf(useRealtime)},
                {"algo", String.valueOf(aggressiveness)},
                {"candle", String.valueOf(useCandles)},
                {"T212", t212ApiToken},
                {"push", pushCutUrlEndpoint},
                {"greed", String.valueOf(greed)},
                {"market", market}
        };
    }

    /**
     * Adds a new notification to the notification list and triggers a system notification
     * on supported platforms (macOS or Windows).
     * <p>
     * This method ensures that:
     * <ul>
     *   <li>Old notifications (over 20 minutes) or for the same symbol are removed to avoid clutter.</li>
     *   <li>Notifications are always added on the Event Dispatch Thread for thread safety.</li>
     *   <li>A system-level notification is sent using terminal-notifier (macOS), SystemTray (Windows), or logs otherwise.</li>
     * </ul>
     *
     * @param title         The notification title (displayed in UI and system notification).
     * @param content       The message body of the notification.
     * @param stockUnitList The related stock units for this notification (can be empty).
     * @param localDateTime The timestamp of the event.
     * @param symbol        The ticker symbol relevant to this notification.
     * @param change        The % change (used for sorting/visual cues).
     * @param config        Additional configuration parameter for Notification object.
     */
    public static void addNotification(String title, String content, List<StockUnit> stockUnitList, LocalDateTime localDateTime, String symbol, double change, int config) {
        // Ensure all notification updates happen on the Swing Event Dispatch Thread for UI safety
        SwingUtilities.invokeLater(() -> {
            LocalDateTime now = LocalDateTime.now(); // Current time to check notification age

            // Loop backwards to safely remove notifications while iterating
            for (int i = notificationListModel.size() - 1; i >= 0; i--) {
                Notification existing = notificationListModel.getElementAt(i);
                long minutesOld = ChronoUnit.MINUTES.between(existing.getLocalDateTime(), now);

                // Remove notifications if:
                // 1. They are older than 20 minutes (stale) OR
                // 2. They refer to the same stock symbol (avoid duplicates)
                if (minutesOld > 20 || existing.getSymbol().equals(symbol)) {
                    notificationListModel.remove(i);
                }
            }

            // Add the new notification to the model/list (appears in the hype panel)
            notificationListModel.addElement(
                    new Notification(title, content, stockUnitList, localDateTime, symbol, change, config)
            );
        });

        // Compose the text body for the system notification
        String notificationContent = String.format("%s\nSymbol: %s\nChange: %.2f%%", content, symbol, change);

        // Determine the user's operating system for platform-specific notification logic
        String osName = System.getProperty("os.name").toLowerCase();

        // --- macOS native notification (requires terminal-notifier to be installed) ---
        if (osName.contains("mac")) {
            try {
                String[] terminalNotifierCommand = {
                        "terminal-notifier",
                        "-title", title,
                        "-message", notificationContent,
                        "-contentImage", Paths.get(System.getProperty("user.dir"), "train.png").toString(),
                        "-sound", "default",
                        "-timeout", "5"
                };
                // Launch system notification as a background process
                new ProcessBuilder(terminalNotifierCommand).start();
            } catch (IOException e) {
                // Log if the notification could not be sent (e.g., terminal-notifier missing)
                System.err.println("Failed to send macOS notification: " + e.getMessage());
            }
        }
        // --- Windows system tray notification ---
        else if (osName.contains("win")) {
            if (SystemTray.isSupported()) {
                // Show a balloon tip notification using the Windows system tray
                displayWindowsNotification(title, notificationContent);
            }
        }
        // --- Other OS: fallback, just print to console (e.g., Linux, unsupported platforms) ---
        else {
            System.out.println("Can't create notification.");
        }

        // Send push notifications to mobile device in order to notify user
        sendPushNotification(title, notificationContent);
    }

    /**
     * Fires a Pushcut notification by spawning a {@code curl} process.
     *
     * @param title the notification title shown in the push banner; must not be {@code null}
     * @param body  the body text below the title; must not be {@code null}
     * @throws NullPointerException if {@code title} or {@code body} is {@code null}
     */
    public static void sendPushNotification(String title, String body) {
        // Validate arguments early to fail fast.
        if (title == null || body == null) {
            throw new NullPointerException("title and body must not be null");
        }

        /*
         * Build the JSON payload that PushCut expects.
         * We escape user‑supplied strings to guarantee valid JSON even if the
         * caller passes quotes, backslashes, or line breaks.
         */
        String json = String.format(
                "{\"title\":\"%s\",\"text\":\"%s\"}",
                escapeJson(title),
                escapeJson(body));

        /*
         * Construct the {@link ProcessBuilder} that will invoke curl:
         *
         *   curl -sS -X POST -H "Content-Type: application/json" -d <json> <endpoint>
         *
         * Options explained:
         *   -sS  : run silently but still print errors
         *   -X   : HTTP method (POST)
         *   -H   : HTTP header
         *   -d   : request body
         */
        ProcessBuilder pb = new ProcessBuilder(
                "curl", "-sS",
                "-X", "POST",
                "-H", "Content-Type: application/json",
                "-d", json,
                pushCutUrlEndpoint);

        // Forward curl’s standard output and error to our own process so it is visible in the console.
        pb.inheritIO();

        try {
            // Start the process (this call is asynchronous)…
            Process process = pb.start();

            // …then wait synchronously for it to finish so we can inspect the exit code.
            if (process.waitFor() != 0) {
                System.err.println("PushCut curl exited with " + process.exitValue());
            }
        } catch (IOException e) {
            // Thrown if the process could not be started (e.g. curl not found).
            System.err.println("Failed to start curl: " + e.getMessage());
        } catch (InterruptedException e) {
            // Restore the interrupted status and record the failure.
            Thread.currentThread().interrupt();
            System.err.println("Waiting for curl was interrupted: " + e.getMessage());
        }
    }

    /**
     * Escapes special characters so that the resulting string can be safely embedded in JSON
     * string literals.
     *
     * @param s the raw, unescaped string
     * @return the escaped version ready for inclusion in a JSON document
     */
    private static String escapeJson(String s) {
        /*
         * The replacements are applied in an order that avoids double‑escaping.
         *
         * 1. Backslash       → double backslash
         * 2. Double quote    → backslash‑escaped quote
         * 3. Line feed (LF)  → \n
         * 4. Carriage return → (strip) – optional optimisation for Windows CRLF
         */
        return s
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "");
    }

    /**
     * Displays a Windows notification in the system tray.
     * <p>
     * This method does nothing if SystemTray is not supported on the current system.
     *
     * @param title   Notification title.
     * @param message Notification body.
     */
    private static void displayWindowsNotification(String title, String message) {
        // Exit immediately if the SystemTray feature is not supported on this platform (e.g., Linux headless)
        if (!SystemTray.isSupported()) return;

        try {
            // Get the system tray instance for the current desktop environment
            SystemTray tray = SystemTray.getSystemTray();

            // Load the tray icon image (relative to working directory)
            Image image = Toolkit.getDefaultToolkit().createImage("train.png");

            // Create a new TrayIcon with the loaded image and a tooltip
            TrayIcon trayIcon = new TrayIcon(image, "Notification");

            // Automatically resize the icon for best fit in the tray
            trayIcon.setImageAutoSize(true);

            // Add the tray icon to the system tray (shows it in the OS tray area)
            tray.add(trayIcon);

            // Show a popup notification balloon with the given title and message
            trayIcon.displayMessage(title, message, TrayIcon.MessageType.INFO);

            // (Note: Icon remains until application exits, or you explicitly remove it.)
        } catch (Exception e) {
            // Print any exceptions (e.g., file not found, security exceptions) for debugging
            e.printStackTrace();
        }
    }

    /**
     * Places a green vertical marker on the provided plot at the specified X position.
     * <p>
     * Used for visually selecting the start point of a range in the chart (e.g., for measuring % change).
     * Clears any previous markers or shaded regions before adding the new one.
     *
     * @param plot      The XYPlot to annotate.
     * @param xPosition The domain (X-axis) value to place the marker at.
     */
    private static void addFirstMarker(XYPlot plot, double xPosition) {
        // Remove any existing markers/shaded region
        if (marker1 != null) {
            plot.removeDomainMarker(marker1);
        }
        if (marker2 != null) {
            plot.removeDomainMarker(marker2);
        }
        if (shadedRegion != null) {
            plot.removeDomainMarker(shadedRegion);
        }

        // Place a new green marker for the first selection
        marker1 = new ValueMarker(xPosition);
        marker1.setPaint(Color.GREEN);                // Visual color: green
        marker1.setStroke(new BasicStroke(1.5f));     // Slightly thick for visibility
        plot.addDomainMarker(marker1);
    }

    /**
     * Adds a second marker and a shaded region between two selected points on the chart.
     * <p>
     * Also calculates the % difference between the two Y values and sets the display label.
     * Colors are green for positive change, red for negative.
     *
     * @param plot The XYPlot to annotate.
     */
    private static void addSecondMarkerAndShade(XYPlot plot) {
        // Remove any previous marker or shaded region
        if (marker2 != null) {
            plot.removeDomainMarker(marker2);
        }
        if (shadedRegion != null) {
            plot.removeDomainMarker(shadedRegion);
        }

        // Compute percentage change between selected Y values
        double percentageDiff = ((point2Y - point1Y) / point1Y) * 100;

        // Set colors based on direction (green for gain, red for loss)
        Color markerColor = (percentageDiff >= 0) ? Color.GREEN : Color.RED;
        Color shadeColor = (percentageDiff >= 0) ? new Color(100, 200, 100, 50) : new Color(200, 100, 100, 50);

        // Add the second marker
        marker2 = new ValueMarker(point2X);
        marker2.setPaint(markerColor);
        marker2.setStroke(new BasicStroke(1.5f));
        plot.addDomainMarker(marker2);

        // Add shaded region between markers
        shadedRegion = new IntervalMarker(Math.min(point1X, point2X), Math.max(point1X, point2X));
        shadedRegion.setPaint(shadeColor);  // Translucent color for region
        plot.addDomainMarker(shadedRegion);

        // Display the calculated percentage change in the UI
        percentageChange.setText(String.format("Percentage Change: %.3f%%", percentageDiff));
    }

    /**
     * Reloads the configuration from disk and applies it throughout the UI.
     * <p>
     * This method:
     * <ul>
     *   <li>Reloads all key-value pairs from config.xml</li>
     *   <li>Repopulates the watchlist and color map</li>
     *   <li>Re-initializes the API connection with the new key</li>
     *   <li>Refreshes all UI components so they reflect the latest settings</li>
     * </ul>
     */
    public static void loadConfig() {
        setValues(); // Reload settings (volume, symbols, etc.)

        loadTable(symbols); // Rebuild watchlist and color map from symbols
        mainUI.refreshAllComponents(gui.getContentPane()); // Force UI to update
        mainDataHandler.InitAPi(apiKey); // Re-initialize API connection (can comment if testing without tokens)

        logTextArea.append("Config reloaded\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
    }

    /**
     * Creates and returns a JButton for removing a selected stock from the watchlist.
     * <p>
     * When clicked, the button removes the selected item from the given {@link JList}, updates the internal
     * {@link #stockColors} map and the {@link #stockListModel}, and regenerates the symbol config string.
     *
     * @param stockList The JList containing the watchlist symbols.
     * @return JButton configured for removing the currently selected symbol.
     */
    @NotNull
    private static JButton getJButton(JList<String> stockList) {
        JButton removeButton = new JButton("-");
        removeButton.addActionListener(e -> {
            // Retrieve the currently selected symbol from the list
            String selectedValue = stockList.getSelectedValue();
            if (selectedValue != null) {
                // Remove the color and the symbol from the models
                stockColors.remove(selectedValue);
                stockListModel.removeElement(selectedValue);

                // Update the symbol string for saving
                symbols = createSymArray();
            }
        });
        return removeButton;
    }

    /**
     * Creates a button for displaying the overview of the currently selected company/stock.
     * <p>
     * When clicked, opens a dialog with a textual overview fetched asynchronously.
     * Handles cases where no stock is selected.
     *
     * @return JButton that shows the company overview dialog on click.
     */
    @NotNull
    private static JButton getOverviewButton() {
        // Create the button for showing company overview
        JButton overviewButton = new JButton("Look at Company Overview");

        // Add a listener to handle button click events
        overviewButton.addActionListener(e -> {
            // If no valid stock is selected, warn the user and do nothing
            if ("-Select a Stock-".equals(selectedStock)) {
                JOptionPane.showMessageDialog(
                        null,
                        "Please select a valid stock.",
                        "Error",
                        JOptionPane.ERROR_MESSAGE
                );
                return;
            }

            // Create a dialog window to display the company overview
            JDialog dialog = new JDialog();
            dialog.setTitle(selectedStock + " - Company Overview"); // Set dialog title
            dialog.setSize(500, 400);                               // Set dialog size
            dialog.setLayout(new BorderLayout());                   // Use border layout
            dialog.setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE); // Dispose dialog on close

            // Panel to hold the content, with padding for aesthetics
            JPanel panel = new JPanel(new BorderLayout());
            panel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

            // Label at the top for the selected stock name
            JLabel titleLabel = new JLabel(selectedStock + " Overview", SwingConstants.CENTER);

            // Text area to show the overview description, initially with a loading message
            JTextArea overviewText = new JTextArea("Fetching company overview...");
            overviewText.setWrapStyleWord(true);     // Wrap words for readability
            overviewText.setLineWrap(true);          // Enable line wrapping
            overviewText.setEditable(false);         // Make text area read-only

            // Put the text area in a scroll pane for long descriptions
            JScrollPane scrollPane = new JScrollPane(overviewText);
            scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
            scrollPane.setPreferredSize(new Dimension(480, 300));

            // Add the title and scrollable text area to the panel
            panel.add(titleLabel, BorderLayout.NORTH);
            panel.add(scrollPane, BorderLayout.CENTER);

            // Add the content panel to the dialog and show it in the center of the screen
            dialog.add(panel);
            dialog.setLocationRelativeTo(null);
            dialog.setVisible(true);

            // Fetch the company overview asynchronously from the API
            mainDataHandler.getCompanyOverview(selectedStock, value -> SwingUtilities.invokeLater(() -> {
                // If the API returned a valid overview, show it; otherwise, show fallback message
                if (value != null && value.getOverview() != null) {
                    overviewText.setText(value.getOverview().getDescription());
                } else {
                    overviewText.setText("No overview available.");
                }
            }));
        });

        // Return the constructed button for use in the UI
        return overviewButton;
    }

    /**
     * Generates a visually distinctive random color (with light hues) for new stocks in the watchlist.
     * <p>
     * Ensures all color channels are in the upper half of the spectrum for better readability on light backgrounds.
     *
     * @return Randomly generated {@link Color}.
     */
    private static Color generateRandomColor() {
        Random rand = new Random();
        int red = rand.nextInt(128) + 128;   // Red in [128, 255]
        int green = rand.nextInt(128) + 128; // Green in [128, 255]
        int blue = rand.nextInt(128) + 128;  // Blue in [128, 255]
        return new Color(red, green, blue);
    }

    /**
     * Refreshes the displayed chart according to the current chart type (candlestick or line),
     * optionally fetching and refilling data if requested.
     * <p>
     * This method updates the chart placeholder with a new chart panel and,
     * if <code>fill</code> is true and a stock is selected, reloads the timeline data
     * (with "outlier" smoothing) and triggers a chart data refresh.
     *
     * @param fill If true, fetches fresh data and fills the chart; if false, only redraws the chart.
     */
    public static void refreshChartType(boolean fill) {
        // Ensure this runs on the Swing Event Dispatch Thread for safe UI updates
        SwingUtilities.invokeLater(() -> {
            // Remove any existing chart from the UI to prepare for the new chart
            chartPlaceholder.removeAll();

            // Create a new chart panel for the currently selected stock,
            // using either candlestick (OHLC) or line chart depending on useCandles
            chartDisplay = createChart(selectedStock + (useCandles ? " OHLC Chart" : " Price Chart"));

            // Add the new chart panel to the placeholder area in the UI
            chartPlaceholder.add(chartDisplay, BorderLayout.CENTER);

            // Make sure the panel layout is recalculated and displayed properly
            chartPlaceholder.revalidate();
            chartPlaceholder.repaint();

            // Only proceed with data fetching and replotting if:
            //  - 'fill' is true (caller wants fresh data)
            //  - A valid stock (not the default "-Select a Stock-") is selected
            if (fill && !selectedStock.contains("lect a Stock")) {
                // Asynchronously fetch historical timeline data for the selected stock
                mainDataHandler.getTimeline(selectedStock, values -> {
                    // Prepare a list of the original close prices to detect outliers (big jumps)
                    List<Double> originalCloses = new ArrayList<>(values.size());
                    for (StockUnit stock : values) {
                        originalCloses.add(stock.getClose());
                    }

                    // Smooth the data: if a close jumps >10% from previous,
                    // set it to previous close to avoid weird spikes in the chart
                    IntStream.range(1, values.size()).parallel().forEach(i -> {
                        double currentOriginalClose = originalCloses.get(i);
                        double previousOriginalClose = originalCloses.get(i - 1);
                        // Detect "outlier" jumps greater than 10%
                        if (Math.abs((currentOriginalClose - previousOriginalClose) / previousOriginalClose) >= 0.1) {
                            // Clamp value too previous to avoid plotting a spike
                            values.get(i).setClose(previousOriginalClose);
                        }
                    });

                    // Now update the global stock data and refresh the chart to show the data
                    // The chart will default to a 3-day window on refresh (choice 9)
                    SwingUtilities.invokeLater(() -> {
                        stocks = values;
                        refreshChartData(9);
                    });
                });
            }
        });
    }

    /**
     * Updates the chart data and time window for the currently selected stock and time range.
     * <p>
     * This method determines if the underlying data series need to be recalculated (based on ticker change,
     * chart type toggle, or time window switch), aggregates or smooths the stock data accordingly,
     * and updates the display for either candlestick (OHLC) or line charts. It also manages all axis
     * and time window settings and re-enables all time range buttons.
     *
     * @param choice The selected time window/range (1=1min, 2=3min, ..., 12=1month).
     */
    public static void refreshChartData(int choice) {
        // Store the user's latest time range selection for future refreshes
        currentTimeRangeChoice = choice;

        // Determine if we need to re-aggregate or recalculate data:
        // - Ticker has changed
        // - Chart type changed (candles ↔ price)
        // - In candlestick mode: window has changed (different aggregation required)
        boolean needReaggregate =
                !selectedStock.equals(currentStockSymbol)   // ticker changed
                        || modeCopy != useCandles           // chart type toggled
                        || (useCandles && choice != lastChoice); // new window in candles mode

        // Only recalculate and redraw the chart if something has actually changed
        if (needReaggregate) {
            // Update mode/ticker tracking
            modeCopy = useCandles;
            currentStockSymbol = selectedStock;
            ohlcSeries.clear();     // Clear candlestick series (if used)
            timeSeries.clear();     // Clear line chart series (if used)
            lastChoice = choice;

            // Set the chart window title with the ticker and chart type
            chartDisplay.getChart().setTitle(selectedStock + (useCandles ? " OHLC Chart" : " Price Chart"));

            try {
                if (stocks != null) {
                    // Work with system time zone for all date operations
                    final ZoneId zone = ZoneId.systemDefault();

                    if (useCandles) {
                        // --- CANDLESTICK AGGREGATION ---
                        // Pick aggregation period based on requested range (e.g. Hour/Day)
                        Class<? extends RegularTimePeriod> periodClass = determineAggregationPeriod(choice);

                        // Compute duration and time range
                        long duration = getDurationMillis(choice);
                        Date endDate = stocks.get(0).getDateDate(); // Latest date in dataset
                        Date startDate = new Date(endDate.getTime() - duration);

                        // Build a cache key for storing aggregation results to speed up re-renders
                        String cacheKey = periodClass.getName() + "_" + endDate.getTime();

                        // Try retrieving from cache (for performance on repeated redraws)
                        Map<RegularTimePeriod, AggregatedStockData> aggregatedData = aggregationCache.get(cacheKey);
                        if (aggregatedData == null) {
                            // If not cached, aggregate the raw stock data into OHLC "candles"
                            aggregatedData = aggregateData(stocks, periodClass, zone, startDate);
                            aggregationCache.put(cacheKey, aggregatedData);
                        }

                        // Always aggregate again to ensure up-to-date data
                        aggregatedData = aggregateData(stocks, periodClass, zone, startDate);

                        // Populate the OHLC series for plotting
                        ohlcSeries.clear();
                        for (Map.Entry<RegularTimePeriod, AggregatedStockData> e : aggregatedData.entrySet()) {
                            AggregatedStockData a = e.getValue();
                            ohlcSeries.add(e.getKey(), a.open, a.high, a.low, a.close);
                        }
                    } else {
                        // --- TIME SERIES (LINE CHART) MODE ---
                        // Prepare and smooth data to avoid visual spikes on the chart
                        List<Map.Entry<Second, Double>> dataPoints = stocks.stream().map(stock -> {
                            // Convert each stock data timestamp to a JFreeChart "Second" object
                            LocalDateTime ldt = LocalDateTime.ofInstant(stock.getDateDate().toInstant(), zone);
                            return new AbstractMap.SimpleEntry<>(
                                    new Second(ldt.getSecond(), ldt.getMinute(), ldt.getHour(),
                                            ldt.getDayOfMonth(), ldt.getMonthValue(), ldt.getYear()),
                                    stock.getClose()
                            );
                        }).collect(Collectors.toList());

                        // Sort by date for reliable charting
                        dataPoints.sort(Map.Entry.comparingByKey());

                        // Smooth the time series: if a single point varies >10% from the previous, clamp it
                        List<Map.Entry<Second, Double>> filteredPoints = new ArrayList<>();
                        Double previousClose = null;
                        for (Map.Entry<Second, Double> entry : dataPoints) {
                            double currentClose = entry.getValue();
                            if (previousClose != null) {
                                double variance = Math.abs((currentClose - previousClose) / previousClose);
                                if (variance > 0.1) currentClose = previousClose;
                            }
                            previousClose = currentClose;
                            filteredPoints.add(new AbstractMap.SimpleEntry<>(entry.getKey(), currentClose));
                        }

                        // Add smoothed values to the JFreeChart TimeSeries for display
                        filteredPoints.forEach(entry -> timeSeries.addOrUpdate(entry.getKey(), entry.getValue()));
                    }

                    // --- DATASET & PLOT REFRESHING ---
                    XYPlot plot = chartDisplay.getChart().getXYPlot();

                    if (useCandles) {
                        // Set the plot dataset to the new OHLC (candlestick) data
                        OHLCSeriesCollection ohlcDataset = new OHLCSeriesCollection();
                        ohlcDataset.addSeries(ohlcSeries);
                        plot.setDataset(ohlcDataset);
                    } else {
                        // Set the plot dataset to the new time series data
                        TimeSeriesCollection tsDataset = new TimeSeriesCollection();
                        tsDataset.addSeries(timeSeries);
                        plot.setDataset(tsDataset);
                    }
                }
            } catch (Exception e) {
                // Log any errors so the user sees what went wrong
                e.printStackTrace();
                logTextArea.append("Error loading data: " + e.getMessage() + "\n");
            }
        }

        // --- AXIS & TIME WINDOW UPDATING ---

        // Get the correct time window for the current choice (e.g. 3 days, 1 week, etc.)
        long duration = getDurationMillis(choice);
        Date endDate;
        if (useCandles) {
            // For OHLC, use the last period in the series, or fallback to now
            endDate = ohlcSeries.getItemCount() > 0
                    ? ohlcSeries.getPeriod(ohlcSeries.getItemCount() - 1).getEnd()
                    : new Date();
        } else {
            // For line chart, use the last time series value, or fallback to now
            endDate = timeSeries.getItemCount() > 0
                    ? timeSeries.getTimePeriod(timeSeries.getItemCount() - 1).getEnd()
                    : new Date();
        }
        // Start date is window duration before end
        Date startDate = new Date(endDate.getTime() - duration);

        // Update the X (date/time) axis to show only the selected window
        XYPlot plot = chartDisplay.getChart().getXYPlot();
        DateAxis axis = (DateAxis) plot.getDomainAxis();
        axis.setRange(startDate, endDate);

        // Update the Y (price) axis range to fit the visible data
        updateYAxisRange(plot, startDate, endDate);

        // Force a visual repaint of the chart
        chartDisplay.repaint();

        // Enable all time window buttons for user interaction
        oneMinutesButton.setEnabled(true);
        threeMinutesButton.setEnabled(true);
        fiveMinutesButton.setEnabled(true);
        tenMinutesButton.setEnabled(true);
        thirtyMinutesButton.setEnabled(true);
        oneHourButton.setEnabled(true);
        fourHourButton.setEnabled(true);
        oneDayButton.setEnabled(true);
        threeDaysButton.setEnabled(true);
        oneWeekButton.setEnabled(true);
        twoWeeksButton.setEnabled(true);
        oneMonthButton.setEnabled(true);
    }

    /**
     * Determines the optimal aggregation period (time bucket) class for chart data, based on the selected time range.
     * <p>
     * For shorter windows (under 5 days), aggregates to the second level for high resolution.
     * For longer windows (≥ 5 days), aggregates to the hour for clarity and performance.
     *
     * @param choice Integer representing the user-selected chart window (e.g., 1-minute, 1-week, etc.).
     * @return The class object for the aggregation period: {@link Hour} or {@link Second}.
     */
    // Helper method to determine aggregation interval
    private static Class<? extends RegularTimePeriod> determineAggregationPeriod(int choice) {
        // Get the window size in milliseconds for the chosen time range
        long duration = getDurationMillis(choice);

        // Use hourly aggregation if the window is 5 days or longer, otherwise use per-second granularity
        if (duration >= TimeUnit.DAYS.toMillis(5)) { // 5 days or more
            return Hour.class;
        } else {
            return Second.class;
        }
    }

    /**
     * Aggregates a list of {@link StockUnit} price data into regular time periods (buckets) such as seconds, hours, or days,
     * computing OHLC (Open, High, Low, Close) values for each period.
     * <p>
     * This is essential for candlestick or bar charting, reducing thousands of raw ticks to a manageable number of visual elements.
     * Each time bucket is represented by a {@link RegularTimePeriod} subclass (e.g., {@link Hour}, {@link Second}).
     *
     * @param stockUnitList List of {@link StockUnit} data points to aggregate.
     * @param periodClass   The class of the time period bucket to use (e.g., {@link Hour}, {@link Second}).
     * @param zone          The time zone to use for period alignment (important for global stocks).
     * @param startDate     Data before this date is ignored; typically this trims the list to the current chart window.
     * @return A sorted map of {@link RegularTimePeriod} to {@link AggregatedStockData} with OHLC data for each period.
     */
    // Data aggregation logic
    private static Map<RegularTimePeriod, AggregatedStockData> aggregateData(
            List<StockUnit> stockUnitList,
            Class<? extends RegularTimePeriod> periodClass, ZoneId zone, Date startDate) {

        // Filter to only include data after the given start date (for the visible chart window)
        List<StockUnit> filteredStocks = stockUnitList
                .stream()
                .filter(stock -> stock.getDateDate().after(startDate))
                .toList();

        // Determine what unit to truncate timestamps to: days, hours, or minutes
        TemporalUnit unit = (periodClass == Day.class) ? ChronoUnit.DAYS
                : (periodClass == Hour.class) ? ChronoUnit.HOURS
                : ChronoUnit.MINUTES; // fallback/default

        // Use the specified time zone and the default locale for RegularTimePeriod creation
        TimeZone timeZone = TimeZone.getTimeZone(zone);
        Locale locale = Locale.getDefault();

        // Temp map: Long = truncated epoch ms for period start, value = running OHLC aggregator for that bucket
        Map<Long, AggregatedStockData> tempMap = new HashMap<>(filteredStocks.size() / 4);

        // For each stock data point in the window...
        for (StockUnit stockUnit : filteredStocks) {
            // Get the epoch ms for the stock's timestamp
            long timestamp = stockUnit.getDateDate().getTime();

            // Convert to ZonedDateTime (for correct time zone handling)
            ZonedDateTime zdt = Instant.ofEpochMilli(timestamp).atZone(zone);

            // Truncate time down to the start of the current bucket (e.g., start of the hour/minute/second)
            ZonedDateTime truncatedZdt = zdt.truncatedTo(unit);
            long bucketStart = truncatedZdt.toInstant().toEpochMilli(); // key for map

            // Try to get the existing aggregate for this bucket, or create if not found
            AggregatedStockData agg = tempMap.get(bucketStart);
            if (agg == null) {
                // New bucket: open, high, low all use first value seen for this period
                agg = new AggregatedStockData();
                agg.open = stockUnit.getOpen();
                agg.high = stockUnit.getHigh();
                agg.low = stockUnit.getLow();
                tempMap.put(bucketStart, agg);
            } else {
                // Existing bucket: update high if this value is higher
                if (stockUnit.getHigh() > agg.high) {
                    agg.high = stockUnit.getHigh();
                }
                // Update low if this value is lower
                if (stockUnit.getLow() < agg.low) {
                    agg.low = stockUnit.getLow();
                }
            }
            // Always set close to the current value, so the last value seen is the closing price
            agg.close = stockUnit.getClose();
        }

        // Now convert our Long-keyed map to a TreeMap keyed by RegularTimePeriod for display
        Map<RegularTimePeriod, AggregatedStockData> aggregatedMap = new TreeMap<>();

        // Sort the bucket starts to ensure period order
        List<Long> sortedBucketStarts = new ArrayList<>(tempMap.keySet());
        Collections.sort(sortedBucketStarts);

        // For each bucket, create the corresponding RegularTimePeriod instance (e.g., Hour, Second, etc.)
        for (Long bucketStart : sortedBucketStarts) {
            Date date = new Date(bucketStart);
            RegularTimePeriod period = RegularTimePeriod.createInstance(periodClass, date, timeZone, locale);
            aggregatedMap.put(period, tempMap.get(bucketStart));
        }

        return aggregatedMap;
    }

    /**
     * Dynamically recalculates and sets the visible range for the Y-axis of the provided chart plot,
     * so that all data points in the given time window [start, end] are visible with a small margin.
     * <p>
     * This works for both OHLC (candlestick) and time series (line chart) modes.
     *
     * @param plot  The XYPlot object whose Y-axis should be adjusted.
     * @param start The start date of the visible window (left edge of X axis).
     * @param end   The end date of the visible window (right edge of X axis).
     */
    private static void updateYAxisRange(XYPlot plot, Date start, Date end) {
        // Retrieve the Y axis object so we can set its range later
        ValueAxis yAxis = plot.getRangeAxis();

        // Initialize min and max trackers to extreme values; these will shrink/expand as we scan
        double minY = Double.MAX_VALUE;
        double maxY = Double.MIN_VALUE;

        // --- HANDLE CANDLESTICK MODE ---
        if (useCandles) {
            // Loop over all OHLC (candlestick) data points in the current data series
            for (int i = 0; i < ohlcSeries.getItemCount(); i++) {
                OHLCItem item = (OHLCItem) ohlcSeries.getDataItem(i);

                // Use the period's end time for window comparison
                Date itemDate = item.getPeriod().getEnd();

                // Only consider points inside the currently visible X window (between start and end)
                if (itemDate.after(start) && itemDate.before(end)) {
                    // Update min/max if this data point is lower/higher than previous seen
                    minY = Math.min(minY, item.getLowValue());
                    maxY = Math.max(maxY, item.getHighValue());
                }
            }
        } else {
            // --- HANDLE SIMPLE TIME SERIES (LINE CHART) MODE ---
            // Loop through all data points in the TimeSeries object
            for (int i = 0; i < timeSeries.getItemCount(); i++) {
                // Each item corresponds to one "period" (e.g., a Second, Minute, etc.)
                Date itemDate = timeSeries.getTimePeriod(i).getEnd();

                // Restrict calculations to data within [start, end] window
                if (itemDate.after(start) && itemDate.before(end)) {
                    // Get the data value for this time period
                    double value = timeSeries.getValue(i).doubleValue();

                    // Update min/max trackers as needed
                    minY = Math.min(minY, value);
                    maxY = Math.max(maxY, value);
                }
            }
        }

        // If we actually found at least one data point in the visible window...
        if (minY != Double.MAX_VALUE && maxY != Double.MIN_VALUE) {
            // ...then set the Y-axis to just outside the actual min/max to give some visual "breathing space"
            // Here, 0.99 = 1% below min, 1.01 = 1% above max
            yAxis.setRange(minY * 0.99, maxY * 1.01);
        }
        // If no points were found, do not touch the axis (prevents axis errors when series is empty)
    }

    /**
     * Converts a chart time range selection (integer) to the corresponding duration in milliseconds.
     * <p>
     * Used for setting the visible window in the chart.
     *
     * @param choice The selected time range (1 = 1min, 2 = 3min, ..., 12 = 1 month).
     * @return The corresponding time span in milliseconds.
     * @throws IllegalArgumentException If the choice is not recognized.
     */
    private static long getDurationMillis(int choice) {
        // Java 17+ switch expression for concise mapping.
        // Each case corresponds to a UI button's time window.
        return switch (choice) {
            case 1 -> TimeUnit.MINUTES.toMillis(1);     // User picked "1 Minute"
            case 2 -> TimeUnit.MINUTES.toMillis(3);     // User picked "3 Minutes"
            case 3 -> TimeUnit.MINUTES.toMillis(5);     // User picked "5 Minutes"
            case 4 -> TimeUnit.MINUTES.toMillis(10);    // User picked "10 Minutes"
            case 5 -> TimeUnit.MINUTES.toMillis(30);    // User picked "30 Minutes"
            case 6 -> TimeUnit.HOURS.toMillis(1);       // User picked "1 Hour"
            case 7 -> TimeUnit.HOURS.toMillis(4);       // User picked "4 Hours"
            case 8 -> TimeUnit.DAYS.toMillis(1);        // User picked "1 Day"
            case 9 -> TimeUnit.DAYS.toMillis(3);        // User picked "3 Days"
            case 10 -> TimeUnit.DAYS.toMillis(7);       // User picked "1 Week"
            case 11 -> TimeUnit.DAYS.toMillis(14);      // User picked "2 Weeks"
            case 12 -> TimeUnit.DAYS.toMillis(30);      // User picked "1 Month" (approx)
            default -> throw new IllegalArgumentException("Invalid time range");
        };
    }

    /**
     * Creates a JFreeChart chart panel based on the current display type.
     * <p>
     * If {@link #useCandles} is true, shows a candlestick (OHLC) chart; otherwise, a time series line chart.
     * Adds a custom renderer and mouse handler for interactivity.
     *
     * @param title The title to display on the chart.
     * @return The fully configured {@link ChartPanel}.
     */
    private static ChartPanel createChart(String title) {
        // Collection for candlestick chart (OHLC data)
        OHLCSeriesCollection ohlcSeriesCollection = new OHLCSeriesCollection();
        ohlcSeriesCollection.addSeries(ohlcSeries);

        // Collection for line chart (price data)
        TimeSeriesCollection timeSeriesCollection = new TimeSeriesCollection();
        timeSeriesCollection.addSeries(timeSeries);

        // Create the chart object using factory, depending on chart mode
        // Candlestick (OHLC) vs. Time series line chart.
        JFreeChart chart = useCandles
                ? ChartFactory.createCandlestickChart(
                title,                // chart title
                "Date",               // X-axis label
                "Price",              // Y-axis label
                ohlcSeriesCollection, // dataset
                true                  // legend
        )
                : ChartFactory.createTimeSeriesChart(
                title,                // chart title
                "Date",               // X-axis label
                "Price",              // Y-axis label
                timeSeriesCollection, // dataset
                true,                 // legend
                true,                 // tooltips
                false                 // URLs
        );

        // Get reference to the plot for further customisation
        XYPlot plot = getXyPlot(chart);

        // Add a default faint green background marker (from Y=0 up)
        // This provides subtle background highlighting
        plot.addRangeMarker(new IntervalMarker(
                0,                      // Y start
                Double.POSITIVE_INFINITY, // Y end (covers all future values)
                new Color(200, 255, 200, 50) // light green, transparent
        ));

        // Build and return the panel, with mouse interactivity attached
        ChartPanel panel = new ChartPanel(chart);
        attachMouseMarker(panel, plot);
        return panel;
    }

    private static XYPlot getXyPlot(JFreeChart chart) {
        XYPlot plot = chart.getXYPlot();

        // If using candlestick mode, set up a custom renderer
        if (useCandles) {
            CandlestickRenderer r = new CandlestickRenderer();

            // Use the "smallest bar" width for best visual density
            r.setAutoWidthMethod(CandlestickRenderer.WIDTHMETHOD_SMALLEST);

            // Set color for up days (price increase) and down days (price decrease)
            r.setUpPaint(Color.GREEN);
            r.setDownPaint(Color.RED);

            // Draw candle borders and show volume overlay bars
            r.setUseOutlinePaint(true);
            r.setDrawVolume(true);

            // Attach renderer to the plot
            plot.setRenderer(r);
        }
        return plot;
    }

    /**
     * Adds mouse click marker logic to a chart panel, allowing the user to select
     * two points for percentage change measurement.
     * <ul>
     *     <li>First click: draws a vertical marker line (green) at the clicked X position.</li>
     *     <li>Second click: draws another marker, shades the region between, and calculates % change.</li>
     *     <li>Resets marker points for next interaction.</li>
     * </ul>
     *
     * @param chartPanel The {@link ChartPanel} to attach the listener to.
     * @param plot       The chart's {@link XYPlot}, required for marker rendering.
     */
    private static void attachMouseMarker(ChartPanel chartPanel, XYPlot plot) {
        // Attach mouse listener for click events
        chartPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                // Translate mouse click from screen to chart (Java2D) coordinates
                Point2D p = chartPanel.translateScreenToJava2D(e.getPoint());

                // Get axes for coordinate conversion
                ValueAxis xAxis = plot.getDomainAxis();
                ValueAxis yAxis = plot.getRangeAxis();

                // Convert the pixel coordinates to data (domain/range) values
                double x = xAxis.java2DToValue(p.getX(), chartPanel.getScreenDataArea(), plot.getDomainAxisEdge());
                double y = yAxis.java2DToValue(p.getY(), chartPanel.getScreenDataArea(), plot.getRangeAxisEdge());

                // If this is the first click (no previous point recorded)
                if (Double.isNaN(point1X)) {
                    // First point selected, set the first marker
                    point1X = x;
                    point1Y = y;
                    addFirstMarker(plot, point1X); // Draw first marker
                } else {
                    // Second click: draw second marker and shaded area
                    point2X = x;
                    point2Y = y;
                    addSecondMarkerAndShade(plot);

                    // Reset for future clicks (user can measure again)
                    point1X = Double.NaN;
                    point1Y = Double.NaN;
                    point2X = Double.NaN;
                    point2Y = Double.NaN;
                }
            }
        });
    }

    /**
     * Creates a button that adds the specified symbol to the watchlist and
     * shows a dialog confirming the action.
     * <p>
     * Will not add if the symbol is already in the list.
     *
     * @param symbol The ticker symbol to add.
     * @param frame  The parent frame for dialog popups.
     * @return A ready-to-use JButton ("Add to Watchlist").
     */
    @NotNull
    private static JButton getJButton(String symbol, JFrame frame) {
        JButton watchlistButton = new JButton("Add to Watchlist");
        watchlistButton.addActionListener(ev -> {
            // Only add if not already present and symbol is non-null
            if (symbol != null && !stockListModel.contains(symbol)) {
                // Add to the visible watchlist
                stockListModel.addElement(symbol);

                // Assign a random color for visual differentiation in UI
                stockColors.put(symbol, generateRandomColor());

                // Update the serialized symbol config for saving/export
                symbols = createSymArray();
            }
            // Show confirmation to the user
            JOptionPane.showMessageDialog(frame, symbol + " added to watchlist.");
        });
        return watchlistButton;
    }

    /**
     * Fetches the latest list of available stocks (instruments) from the Trading212 live API,
     * parses the returned JSON, and populates the {@link #nameToData} map for quick lookups.
     * <p>
     * This method performs the following steps:
     * <ul>
     *   <li>Makes a HTTP GET request to Trading212's instrument metadata endpoint (requires a valid API token).</li>
     *   <li>Parses the response body as JSON, iterating over each instrument.</li>
     *   <li>For each instrument, extracts the user-friendly short name, ticker symbol, and maximum open quantity.</li>
     *   <li>Populates the {@link #nameToData} map with a mapping from short name to {@link TickerData}.</li>
     *   <li>Skips any incomplete or empty instrument entries.</li>
     * </ul>
     *
     * @throws Exception If the HTTP request fails, response is not valid JSON, or any network errors occur.
     */
    public static void fetchTickerMap() throws Exception {
        // Build a modern HTTP client instance (Java 11+)
        var httpClient = HttpClient.newBuilder().build();
        // The Trading212 endpoint for instrument metadata (stocks, indices, etc.)
        var url = "https://live.trading212.com/api/v0/equity/metadata/instruments";

        // Build the HTTP GET request, including required Authorization header (your Trading212 token)
        var request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Authorization", t212ApiToken)
                .GET()
                .build();

        // Execute the request and extract the JSON response as a String
        String body = httpClient.send(request, HttpResponse.BodyHandlers.ofString()).body();

        // Attempt to parse the API response body into a JSON array of instruments
        JSONArray instruments;
        try {
            // Parse the string into a JSONArray for structured access to each instrument object
            instruments = new JSONArray(body);
        } catch (JSONException e) {
            // If parsing fails (invalid JSON, wrong structure, etc.), print the raw response for debugging
            System.out.println("Failed to parse response as JSONArray. Raw response:");
            System.out.println(body);
            instruments = new JSONArray();
        }

        // Iterate through every instrument (stock, index, etc.) in the JSON array
        for (int i = 0; i < instruments.length(); i++) {
            JSONObject obj = instruments.getJSONObject(i);

            // Extract relevant data (maybe empty/missing)
            String shortName = obj.optString("shortName");     // User-friendly name (e.g., "Apple Inc.")
            String ticker = obj.optString("ticker");           // Trading symbol (e.g., "AAPL")
            int maxOpenQuantity = obj.optInt("maxOpenQuantity"); // Broker limit

            // Only add valid entries (no empty names/tickers)
            if (!shortName.isEmpty() && !ticker.isEmpty()) {
                nameToData.put(shortName, new TickerData(ticker, maxOpenQuantity));
            }
        }
    }

    /**
     * Retrieves {@link TickerData} for a given stock name (user-friendly name, not ticker symbol).
     * If the name is not found, returns a default TickerData object (symbol = name, maxOpenQuantity = max int).
     * <p>
     * This provides a safe fallback if the stock was not in the fetched map.
     *
     * @param name The display name (e.g., "Apple Inc.") of the stock.
     * @return The corresponding {@link TickerData} object, or a fallback if not found.
     */
    public static TickerData getTickerData(String name) {
        // Use Map.getOrDefault to avoid nulls (fallback: symbol=name, quantity=max)
        return nameToData.getOrDefault(name, new TickerData(name, Integer.MAX_VALUE));
    }

    /**
     * Retrieves the trading ticker symbol (e.g., "AAPL") for a given stock name.
     * If not found, returns the original name string.
     *
     * @param name The display name (e.g., "Apple Inc.").
     * @return The trading symbol (e.g., "AAPL"), or the name if missing.
     */
    public static String getTicker(String name) {
        // Just return the ticker property from TickerData (see above method for fallback logic)
        return getTickerData(name).ticker;
    }

    /**
     * Handles a user selection of a stock/ticker, fetching and displaying the corresponding data on the UI.
     * <p>
     * When a user clicks a stock (from a search or watchlist), this method:
     * <ul>
     *   <li>Sets the {@link #selectedStock} to the chosen symbol (uppercase, trimmed).</li>
     *   <li>Asynchronously fetches historical price/timeline data using {@link mainDataHandler#getTimeline}.</li>
     *   <li>Smooths outliers in the close price (>10% jump) for better chart visualisation.</li>
     *   <li>Updates the UI/chart on the Event Dispatch Thread to show the latest smoothed data.</li>
     *   <li>Starts real-time price updates for the newly selected stock.</li>
     *   <li>Handles all errors gracefully by printing stack traces for debugging.</li>
     * </ul>
     *
     * @param symbol The symbol (e.g., "AAPL", "GOOG", "TSLA") selected by the user.
     */
    public void handleStockSelection(String symbol) {
        try {
            // Normalize input: always uppercase and no leading/trailing spaces
            selectedStock = symbol.toUpperCase().trim();

            // Asynchronously fetch the historical timeline for the stock
            mainDataHandler.getTimeline(selectedStock, values -> {
                // Build a list of original close prices to detect "jumps"
                List<Double> originalCloses = new ArrayList<>(values.size());
                for (StockUnit stock : values) {
                    originalCloses.add(stock.getClose());
                }

                // Smoothing: For every price, if the jump from the previous is >10%,
                // clamp it to the previous value to avoid chart spikes/artifacts
                IntStream.range(1, values.size()).parallel().forEach(i -> {
                    double currentOriginalClose = originalCloses.get(i);
                    double previousOriginalClose = originalCloses.get(i - 1);
                    if (Math.abs((currentOriginalClose - previousOriginalClose) / previousOriginalClose) >= 0.1) {
                        values.get(i).setClose(previousOriginalClose);
                    }
                });

                // After smoothing, update global stock data and refresh the chart (UI-safe thread)
                SwingUtilities.invokeLater(() -> {
                    stocks = values;
                    refreshChartData(9); // Default to 3-day view
                });
            });

            // Start real-time updates for this stock (on UI thread)
            SwingUtilities.invokeLater(this::startRealTimeUpdates);

        } catch (Exception ex) {
            // Any exceptions (network, JSON, etc.) are logged for debugging
            ex.printStackTrace();
        }
    }

    /**
     * Builds and returns the main panel for symbol management, including:
     * <ul>
     *     <li>Stock symbol watchlist (with color coding and remove button)</li>
     *     <li>Dynamic search field with real-time suggestions and add button</li>
     *     <li>Asynchronous updating of stock info, price data, and news feed upon selection</li>
     * </ul>
     *
     * <p>
     * This panel is designed to let users:
     * <ul>
     *   <li>See all stocks in their watchlist</li>
     *   <li>Remove stocks from the watchlist with a "-" button</li>
     *   <li>Search and add new stocks to the watchlist using the dynamic search system</li>
     *   <li>Upon selecting a stock, fetch and display stock info, timeline, and news asynchronously</li>
     * </ul>
     *
     * <p>
     * Layout: BorderLayout with search field (NORTH), scrollable stock list (CENTER),
     * button panel (+/-) (SOUTH), and suggestion list (EAST).
     *
     * @return Fully constructed JPanel containing the symbol management UI
     */
    public JPanel createSymbolPanel() {
        // Main container using BorderLayout for logical placement of subcomponents
        JPanel panel = new JPanel(new BorderLayout());
        panel.setPreferredSize(new Dimension(250, 0)); // Fixed width for sidebar appearance

        // ---- SEARCH FIELD (TOP) ----
        // Allows user to type to search for stocks; suggestions appear in the right-hand list
        searchField = new JTextField();
        searchField.setBorder(BorderFactory.createTitledBorder("Search"));

        // ---- SEARCH RESULTS LIST (EAST) ----
        // Model for holding the suggestions found by typing in the search field
        DefaultListModel<String> searchListModel = new DefaultListModel<>();
        // List component to display current search results (suggested symbols)
        JList<String> searchList = new JList<>(searchListModel);
        searchList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        JScrollPane searchScrollPane = new JScrollPane(searchList);
        searchScrollPane.setPreferredSize(new Dimension(125, 0)); // Narrow vertical list

        // ---- WATCHLIST (CENTER) ----
        // This is the persistent user watchlist, displayed in a list with colors
        stockListModel = new DefaultListModel<>();
        JList<String> stockList = new JList<>(stockListModel);
        stockList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // This map keeps the background color associated with each symbol for color-coding
        stockColors = new HashMap<>();

        // Custom renderer: colors each list cell according to its assigned color,
        // adds a border, and sets text color for selected items
        stockList.setCellRenderer((list, value, index, isSelected, cellHasFocus) -> {
            JLabel label = new JLabel(value, JLabel.CENTER); // Center-align the text
            // Get the background color for the symbol, fallback to gray if absent
            Color fixedColor = stockColors.getOrDefault(value, Color.LIGHT_GRAY);
            label.setOpaque(true);
            label.setBackground(fixedColor);
            // Black border (rounded corners)
            label.setBorder(BorderFactory.createLineBorder(Color.BLACK, 2, true));
            // On selection, change text color for better contrast, but keep bg unchanged
            label.setForeground(isSelected ? Color.WHITE : Color.BLACK);
            return label;
        });

        // ---- WATCHLIST SELECTION HANDLER ----
        // When the user selects a stock in their watchlist, fetch all its info/news/data in parallel
        stockList.addListSelectionListener(e -> {
            // This check prevents duplicate/extra events when the selection changes
            if (!e.getValueIsAdjusting()) { // Only react when selection is finalized, not as the mouse moves
                try {
                    // Set the selected stock symbol, normalized for safety.
                    // Always uppercase and trimmed to avoid mismatches from extra spaces or casing.
                    selectedStock = stockList.getSelectedValue().toUpperCase().trim();
                } catch (Exception ignored) {
                    // If nothing is selected, or getSelectedValue() is null, just skip the rest silently
                }

                // 1. === Fetch and display basic info (open, high, low, volume, etc.) ===
                // This updates the "Stock Information" panel with the latest numbers.
                mainDataHandler.getInfoArray(selectedStock, values -> {
                    if (values != null && values.length >= 9) {
                        // Defensive: Replace any null elements with 0.0 to prevent UI errors or NPE
                        Arrays.setAll(values, i -> values[i] == null ? 0.00 : values[i]);

                        // Immediately update the stock info labels in the UI with this array
                        updateStockInfoLabels(
                                values[0], values[1], values[2], values[3],
                                values[4], values[5], values[6], values[7], values[8]
                        );
                    } else {
                        // If backend returns insufficient data, log the error in the on-screen log
                        SwingUtilities.invokeLater(() -> {
                            logTextArea.append("Received null or incomplete data.\n");
                            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
                        });
                        // Throw a runtime exception to catch issues during testing (optional)
                        throw new RuntimeException("Received null or incomplete data.");
                    }
                });

                // 2. === Fetch and display price timeline data (for the main chart) ===
                // Timeline is fetched and "smoothed": outlier points with >10% jump from previous are clamped for cleaner charts.
                mainDataHandler.getTimeline(selectedStock, values -> {
                    // Gather the original closing prices in a list for reference
                    List<Double> originalCloses = new ArrayList<>(values.size());
                    for (StockUnit stock : values)
                        originalCloses.add(stock.getClose());

                    // Parallel smoothing: For each value, if >10% away from previous, set to previous
                    IntStream.range(1, values.size()).parallel().forEach(i -> {
                        double currentOriginalClose = originalCloses.get(i);
                        double previousOriginalClose = originalCloses.get(i - 1);
                        if (Math.abs((currentOriginalClose - previousOriginalClose) / previousOriginalClose) >= 0.1) {
                            values.get(i).setClose(previousOriginalClose); // Outlier squashed
                        }
                    });

                    // On the UI thread: update the current stock data and refresh the chart for the new symbol
                    SwingUtilities.invokeLater(() -> {
                        stocks = values;
                        refreshChartData(9); // Default to 3-day window
                    });
                });

                // 3. === Fetch and display latest news headlines for the selected symbol ===
                mainDataHandler.receiveNews(selectedStock, values -> {
                    // Sort values by highest relevance score
                    values.sort((a, b) -> Double.compare(b.getSentimentForTicker(selectedStock).getRelevanceScore(), a.getSentimentForTicker(selectedStock).getRelevanceScore()));

                    // All UI updates must run on the Event Dispatch Thread (EDT)
                    SwingUtilities.invokeLater(() -> {
                        NewsListModel.clear(); // Wipe previous news (for another stock)
                        for (com.crazzyghost.alphavantage.news.response.NewsResponse.NewsItem value : values) {
                            // Add each headline and summary as a News object
                            addNews(value.getTitle(), value.getSummary(), value.getUrl(), value.getSentimentForTicker(selectedStock));
                        }
                    });
                });

                // 4. === If "real-time" mode is enabled, start polling live data for this stock ===
                if (useRealtime) {
                    // Only one live update thread should be active for the currently selected symbol.
                    SwingUtilities.invokeLater(this::startRealTimeUpdates);
                }
            }
        });

        // ---- SCROLLABLE WATCHLIST PANEL (CENTER) ----
        JScrollPane stockScrollPane = new JScrollPane(stockList);

        // ---- BUTTON PANEL (BOTTOM) ----
        // Contains "-" (remove from watchlist) and "+" (add from search) buttons
        JButton removeButton = getJButton(stockList); // Remove selected
        JButton addButton = new JButton("+");         // Add selected from search
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        buttonPanel.add(removeButton);
        buttonPanel.add(addButton);

        // Handler for "+" add button
        addButton.addActionListener(e -> {
            String selectedSymbol = searchList.getSelectedValue();
            // Only add if a symbol is selected and it isn't already in the watchlist
            if (selectedSymbol != null && !stockListModel.contains(selectedSymbol)) {
                stockListModel.addElement(selectedSymbol);           // Add to list model
                stockColors.put(selectedSymbol, generateRandomColor()); // Assign a color
                symbols = createSymArray();                         // Update symbol array config
            }
        });

        // ---- ASSEMBLE PANEL LAYOUT ----
        panel.add(searchField, BorderLayout.NORTH);        // Top: search field
        panel.add(stockScrollPane, BorderLayout.CENTER);   // Center: watchlist
        panel.add(buttonPanel, BorderLayout.SOUTH);        // Bottom: +/-
        panel.add(searchScrollPane, BorderLayout.EAST);    // Right: search suggestions

        // Add border with title for clarity
        panel.setBorder(BorderFactory.createTitledBorder("Stock Symbols"));

        // ---- SEARCH FIELD LIVE SUGGESTION LOGIC ----
        // Whenever the search box is changed, trigger an async symbol lookup for live suggestions
        searchField.getDocument().addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                updateSearchList();
            }

            @Override
            public void removeUpdate(DocumentEvent e) {
                updateSearchList();
            }

            @Override
            public void changedUpdate(DocumentEvent e) {
                updateSearchList();
            }

            // Core logic for updating search suggestions list
            private void updateSearchList() {
                String searchText = searchField.getText().trim().toUpperCase();
                searchListModel.clear(); // Always clear before updating

                // If not empty, look for matches asynchronously using mainDataHandler
                if (!searchText.isEmpty()) {
                    // Use the async findMatchingSymbols method with a callback
                    mainDataHandler.findMatchingSymbols(searchText, new mainDataHandler.SymbolSearchCallback() {
                        @Override
                        public void onSuccess(List<String> matchedSymbols) {
                            // Always update list model on the EDT
                            SwingUtilities.invokeLater(() -> searchListModel.addAll(matchedSymbols));
                        }

                        @Override
                        public void onFailure(Exception e) {
                            // Log/print errors on EDT as well
                            SwingUtilities.invokeLater(e::printStackTrace);
                        }
                    });
                }
            }
        });

        // Finally, return the constructed panel
        return panel;
    }

    /**
     * Starts live (real-time) updating of the main chart and notification charts,
     * periodically polling for new data every second.
     * <p>
     * This method:
     * <ul>
     *     <li>Schedules a background task to fetch the latest data for the currently selected stock,
     *         updating the chart and all visible data structures.</li>
     *     <li>Updates the main chart and the charts inside any active notifications,
     *         so that all visualizations are always current.</li>
     *     <li>Makes all UI changes on the Event Dispatch Thread using {@link SwingUtilities#invokeLater(Runnable)}
     *         for thread safety.</li>
     *     <li>Handles all error conditions gracefully, reporting failures in the logging panel.</li>
     * </ul>
     * <b>Note:</b> This function only starts polling if the global <code>useRealtime</code> flag is set to true.
     */
    public void startRealTimeUpdates() {
        // Create a new scheduled executor that will handle the polling loop for notifications (not the main chart)
        final ScheduledExecutorService scheduledExecutor = Executors.newScheduledThreadPool(1);

        // Only set up polling if real-time updates are enabled in settings
        if (useRealtime) {
            // 1. Main chart update: Poll for new data every second, using a global (singleton) executor

            // The executorService is shared globally across the application.
            // This scheduled task will continuously:
            // - Fetch the latest "tick" for the selected stock.
            // - If new data is available, update all chart datasets (TimeSeries, OHLCSeries).
            // - Redraw the chart panel in the GUI.
            executorService.scheduleAtFixedRate(() -> {
                // Double-check flag in case user disables real-time while running
                if (useRealtime) {
                    // Fetch the latest data point for the selected stock (async API call)
                    mainDataHandler.getRealTimeUpdate(selectedStock, value -> {
                        // Only proceed if we actually got a non-null result
                        if (value != null) {
                            // Insert the new stock data at the *start* of the data list,
                            // ensuring the newest tick is always first for visual updates
                            stocks.add(0, new StockUnit.Builder()
                                    .open(value.getOpen())
                                    .high(value.getHigh())
                                    .low(value.getLow())
                                    .close(value.getClose())
                                    .adjustedClose(value.getClose())
                                    .volume(value.getVolume())
                                    .dividendAmount(0)
                                    .splitCoefficient(0)
                                    .time(value.getTimestamp())
                                    .build());

                            // All chart and UI updates must run on the Swing EDT for safety
                            SwingUtilities.invokeLater(() -> {
                                try {
                                    // Parse the server timestamp into a Java Date for plotting
                                    Date date = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(value.getTimestamp());

                                    // Get the current plot and X-axis for auto-scrolling logic
                                    XYPlot plot = chartDisplay.getChart().getXYPlot();
                                    DateAxis xAxis = (DateAxis) plot.getDomainAxis();

                                    // Calculate current time window in milliseconds (e.g., 5 min, 1 hr)
                                    long window = getDurationMillis(currentTimeRangeChoice);

                                    /*
                                     * If the user hasn't zoomed in or out (auto-range is true),
                                     * scroll the chart window forward to always show the latest data.
                                     * This acts like a rolling window—otherwise, respect user zoom.
                                     */
                                    if (xAxis.isAutoRange()) {
                                        xAxis.setAutoRange(true);            // re-apply auto-range
                                        xAxis.setFixedAutoRange(window);     // window width in ms
                                        Date lower = new Date(date.getTime() - window);
                                        updateYAxisRange(plot, lower, date); // auto-adjust y-axis range for window
                                    }

                                    // Update the time series dataset with new close price for this second
                                    double latestClose = value.getClose();
                                    timeSeries.addOrUpdate(new Second(date), latestClose);

                                    // Add the new data to the OHLC series as a new candle/bar.
                                    // If there is a duplicate time or invalid, this may throw—so ignore errors.
                                    try {
                                        ohlcSeries.add(new OHLCItem(
                                                new Second(date),
                                                value.getOpen(),
                                                value.getHigh(),
                                                value.getLow(),
                                                value.getClose()
                                        ));
                                    } catch (Exception ignored) {
                                        // Sometimes throws if duplicate key—ignore for now
                                    }

                                    // Request the chartPanel to repaint itself with the new data
                                    chartPanel.repaint();
                                } catch (Exception e) {
                                    // Any error parsing data or updating UI is logged for debugging
                                    e.printStackTrace();
                                    logTextArea.append("Error updating chart: " + e.getMessage() + "\n");
                                    logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
                                }
                            });
                        } else {
                            // If no new data was returned, log for debugging
                            logTextArea.append("No real-time updates available\n");
                            logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
                        }
                    });
                }
            }, 0, 1, TimeUnit.SECONDS); // Start immediately, repeat every 1 second

            // 2. Notification panel update: Update each notification's chart with live data

            // This scheduled task runs in its own thread and:
            // - Iterates over all active notifications in notificationListModel.
            // - For each, fetches the current tick for the associated stock symbol.
            // - Updates the notification's chart panel and datasets accordingly.
            scheduledExecutor.scheduleAtFixedRate(() -> {
                try {
                    // Only run if there are active notifications
                    if (!notificationListModel.isEmpty()) {
                        for (int i = 0; i < notificationListModel.size(); i++) {
                            Notification notification = notificationListModel.getElementAt(i);

                            // For each notification, fetch a real-time tick for that symbol
                            mainDataHandler.getRealTimeUpdate(notification.getSymbol(), value -> {
                                // Skip if no update available for this notification's symbol
                                if (value == null) return;

                                // All updates must happen on Swing's UI thread
                                SwingUtilities.invokeLater(() -> {
                                    try {
                                        // Parse the time and build a time period for plotting
                                        Date newDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(value.getTimestamp());
                                        Second period = new Second(newDate);

                                        // Get the starting date for this notification's time series window
                                        Date start = notification.getTimeSeries().getDataItem(0).getPeriod().getStart();

                                        // Add a new candle/bar to the notification's OHLC series if it's new
                                        OHLCSeries ohlcSeries = notification.getOHLCSeries();
                                        if (ohlcSeries.getItemCount() == 0 || !ohlcSeries.getPeriod(ohlcSeries.getItemCount() - 1).equals(period)) {
                                            ohlcSeries.add(new OHLCItem(
                                                    period,
                                                    value.getOpen(),
                                                    value.getHigh(),
                                                    value.getLow(),
                                                    value.getClose()
                                            ));
                                        }

                                        // Add/update the closing value for the time series in the notification
                                        TimeSeries series = notification.getTimeSeries();
                                        series.addOrUpdate(period, value.getClose());

                                        // Only attempt to update chart if it's currently visible
                                        ChartPanel chartPanel = notification.getChartPanel();
                                        if (chartPanel != null) {
                                            // Obtain the chart plot (where axes and data are drawn)
                                            XYPlot plot = chartPanel.getChart().getXYPlot();

                                            // The X-axis for time (dates) in this chart
                                            DateAxis axis = (DateAxis) plot.getDomainAxis();

                                            // === Move X-axis window ===
                                            // Adjust the horizontal axis so that the chart view always includes the newest point
                                            // The window will show everything from 'start' to 'newDate'
                                            axis.setRange(start, newDate);

                                            // === Prepare for Y-axis dynamic scaling ===

                                            // Initialize trackers for the minimum and maximum values to extremely wide limits
                                            // We'll update these as we scan through the data to fit just the relevant window
                                            double minY = Double.MAX_VALUE;
                                            double maxY = -Double.MAX_VALUE;

                                            // === Scan through all close values in the notification's TimeSeries ===
                                            // We want to find the lowest and highest close price between 'start' and 'newDate'
                                            for (int j = 0; j < series.getItemCount(); j++) {
                                                Date date = series.getTimePeriod(j).getEnd();
                                                // Only consider points inside the current X-axis window (not before 'start' and not after 'newDate')
                                                if (!date.before(start) && !date.after(newDate)) {
                                                    double closeValue = series.getValue(j).doubleValue();
                                                    // Update minimum and maximum close values found so far
                                                    minY = Math.min(minY, closeValue);
                                                    maxY = Math.max(maxY, closeValue);
                                                }
                                            }

                                            // === Scan through all OHLC values in the same window ===
                                            // OHLC (candles) give extra info: check for highest high and lowest low
                                            for (int k = 0; k < ohlcSeries.getItemCount(); k++) {
                                                OHLCItem item = (OHLCItem) ohlcSeries.getDataItem(k);
                                                Date itemDate = item.getPeriod().getEnd();
                                                // Again, only consider OHLC data inside the visible window
                                                if (!itemDate.before(start) && !itemDate.after(newDate)) {
                                                    // Update min and max if this OHLC bar sets a new high or low
                                                    minY = Math.min(minY, item.getLowValue());
                                                    maxY = Math.max(maxY, item.getHighValue());
                                                }
                                            }

                                            // === Update the Y-axis range if we found at least one valid value ===
                                            if (minY != Double.MAX_VALUE && maxY != -Double.MAX_VALUE) {
                                                ValueAxis yAxis = plot.getRangeAxis();
                                                // Add a 10% margin above and below for better readability (no data right at the edge)
                                                yAxis.setRange(
                                                        minY - (maxY - minY) * 0.1,      // Lower bound: a bit below the min
                                                        maxY + (maxY - minY) * 0.1 + 0.1 // Upper bound: a bit above the max (+0.1 in case of flat range)
                                                );
                                            }

                                            // === Redraw chart with the newly updated axis ranges and datasets ===
                                            chartPanel.repaint();
                                        }
                                    } catch (Exception ex) {
                                        // Log any unexpected error during the notification update
                                        ex.printStackTrace();
                                    }
                                });
                            });
                        }
                    }
                } catch (Exception ex) {
                    // If something unexpected goes wrong, log it for debugging
                    ex.printStackTrace();
                }
            }, 0, 1, TimeUnit.SECONDS); // Start immediately, repeat every 1 second
        }
    }

    /**
     * Constructs the central chart tools panel containing the chart itself, time range controls,
     * company news section, and a summary of stock data.
     * <p>
     * The layout is organized as follows:
     * <ul>
     *     <li><b>Top row</b>: The chart (with selectable timeframes) and a company news feed + company overview button</li>
     *     <li><b>Bottom row</b>: Quick-glance stock data: open, high, low, volume, ratios, and percentage change</li>
     * </ul>
     * Every component is built in code, enabling dynamic UI updates as data or user selection changes.
     *
     * @return The fully constructed JPanel for insertion into the main window.
     */
    public JPanel createChartToolPanel() {
        // Main panel for this section (holds both chart/news and stock info panels)
        JPanel mainPanel = new JPanel(new BorderLayout());

        // ========== FIRST ROW: CHART + NEWS ==========
        JPanel firstRowPanel = new JPanel(new BorderLayout());

        // ---- LEFT: The Chart and Time Range Buttons ----
        chartPanel = new JPanel(new BorderLayout());

        // Create a panel for all time range selection buttons
        JPanel buttonPanel = new JPanel();
        // 2 rows, 5 columns, with 5px spacing in both directions
        buttonPanel.setLayout(new GridLayout(2, 5, 5, 5));

        // Create each time window button for the chart (labels show the range)
        oneMinutesButton = new JButton("1 Minute");
        threeMinutesButton = new JButton("3 Minutes");
        fiveMinutesButton = new JButton("5 Minutes");
        tenMinutesButton = new JButton("10 Minutes");
        thirtyMinutesButton = new JButton("30 Minutes");
        oneHourButton = new JButton("1 Hour");
        fourHourButton = new JButton("4 Hours");
        oneDayButton = new JButton("1 Day");
        threeDaysButton = new JButton("3 Days");
        oneWeekButton = new JButton("1 Week");
        twoWeeksButton = new JButton("2 Weeks");
        oneMonthButton = new JButton("1 Month");

        // Disable all range buttons by default, will be enabled when data loads
        oneMinutesButton.setEnabled(false);
        threeMinutesButton.setEnabled(false);
        fiveMinutesButton.setEnabled(false);
        tenMinutesButton.setEnabled(false);
        thirtyMinutesButton.setEnabled(false);
        oneHourButton.setEnabled(false);
        fourHourButton.setEnabled(false);
        oneDayButton.setEnabled(false);
        threeDaysButton.setEnabled(false);
        oneWeekButton.setEnabled(false);
        twoWeeksButton.setEnabled(false);
        oneMonthButton.setEnabled(false);

        // Add all range buttons to the time selector grid (order is left-to-right, top-to-bottom)
        buttonPanel.add(oneMinutesButton);
        buttonPanel.add(threeMinutesButton);
        buttonPanel.add(fiveMinutesButton);
        buttonPanel.add(tenMinutesButton);
        buttonPanel.add(thirtyMinutesButton);
        buttonPanel.add(oneHourButton);
        buttonPanel.add(fourHourButton);
        buttonPanel.add(oneDayButton);
        buttonPanel.add(threeDaysButton);
        buttonPanel.add(oneWeekButton);
        buttonPanel.add(twoWeeksButton);
        buttonPanel.add(oneMonthButton);

        // Associate each button with the correct chart data refresh (by "choice" index)
        oneMinutesButton.addActionListener(e -> refreshChartData(1));
        threeMinutesButton.addActionListener(e -> refreshChartData(2));
        fiveMinutesButton.addActionListener(e -> refreshChartData(3));
        tenMinutesButton.addActionListener(e -> refreshChartData(4));
        thirtyMinutesButton.addActionListener(e -> refreshChartData(5));
        oneHourButton.addActionListener(e -> refreshChartData(6));
        fourHourButton.addActionListener(e -> refreshChartData(7));
        oneDayButton.addActionListener(e -> refreshChartData(8));
        threeDaysButton.addActionListener(e -> refreshChartData(9));
        oneWeekButton.addActionListener(e -> refreshChartData(10));
        twoWeeksButton.addActionListener(e -> refreshChartData(11));
        oneMonthButton.addActionListener(e -> refreshChartData(12));

        // Panel that will hold the live chart (OHLC or line) - all chart changes are drawn here
        chartPlaceholder = new JPanel(new BorderLayout());
        // Give it a border for visual clarity and set a preferred size for appearance
        chartPlaceholder.setBorder(BorderFactory.createLineBorder(Color.BLACK));
        chartPlaceholder.setPreferredSize(new Dimension(600, 400));

        // Create a special JPanel that automatically replaces previous content when adding new chart panels
        chartPlaceholder = new JPanel(new BorderLayout()) {
            @Override
            public Component add(Component comp) {
                // Only ever show one chart at a time
                removeAll();
                return super.add(comp);
            }
        };

        // -- Chart Data Initialization --
        // Prepare new empty TimeSeries and OHLCSeries so the chart is always initialized even with no data yet
        timeSeries = new TimeSeries(selectedStock + " price");
        ohlcSeries = new OHLCSeries(selectedStock + " price");
        chartDisplay = createChart(selectedStock + " price Chart");

        // Style the chart panel section
        chartPanel.setBorder(BorderFactory.createTitledBorder("Stock price Chart"));
        chartPanel.add(buttonPanel, BorderLayout.NORTH);      // Timeframe buttons at the top
        chartPanel.add(chartPlaceholder, BorderLayout.CENTER); // The chart itself (in the center)

        // ---- RIGHT: The Company News Feed ----
        // News data will be added to this model by news-fetching code elsewhere
        NewsListModel = new DefaultListModel<>();
        JScrollPane newsScrollPane = getNewsScrollPane(); // Method builds a scrollable news list

        // Give the news section a titled border for clarity
        newsScrollPane.setBorder(BorderFactory.createTitledBorder("Company News"));

        // Get the JList from the scroll pane to customize cell appearance
        JList<News> newsList = (JList<News>) newsScrollPane.getViewport().getView();

        // -- Custom rendering for news items
        newsList.setCellRenderer(new NotificationRenderer());

        // -- Set a fixed row height so all news items fit their two-line HTML display
        JLabel dummyLabel = new JLabel("<html>Line1<br>Line2</html>");
        dummyLabel.setFont(newsList.getFont());
        int rowHeight = dummyLabel.getPreferredSize().height + 5; // Extra 5px padding
        newsList.setFixedCellHeight(rowHeight);

        // ---- News Panel Container (news + overview button) ----
        JPanel newsContainerPanel = new JPanel(new BorderLayout());
        newsContainerPanel.setPreferredSize(new Dimension(200, 400));

        // Adjust news scroll area for the space required by the button
        newsScrollPane.setPreferredSize(new Dimension(200, 380));

        // Button at the bottom of the news pane, opens the company overview dialog
        JButton overviewButton = getOverviewButton();

        // Compose the news section: news list in the center, overview button below
        newsContainerPanel.add(newsScrollPane, BorderLayout.CENTER);
        newsContainerPanel.add(overviewButton, BorderLayout.SOUTH);

        // Add the chart and news containers to the first row of the main panel
        firstRowPanel.add(chartPanel, BorderLayout.CENTER);         // Chart & controls (left)
        firstRowPanel.add(newsContainerPanel, BorderLayout.EAST);   // News section (right)

        // ========== SECOND ROW: STOCK DATA SUMMARY ==========
        // This row contains a 4-column summary of current stock data (shown below the chart/news)
        JPanel secondRowPanel = new JPanel(new GridLayout(1, 4)); // 1 row, 4 columns

        // ----- First Column: Open, High, Low -----
        // Create a panel for the "open", "high", and "low" values, stacked vertically (3 rows, 1 column)
        JPanel openHighLowPanel = new JPanel(new GridLayout(3, 1));

        // Create labels for each statistic with default text
        openLabel = new JLabel("Open: ");
        highLabel = new JLabel("High: ");
        lowLabel = new JLabel("Low: ");

        // Add each label to its own row in the first column panel
        openHighLowPanel.add(openLabel);
        openHighLowPanel.add(highLabel);
        openHighLowPanel.add(lowLabel);

        // ----- Second Column: Volume, P/E, PEG -----    // This column shows the trading volume, P/E ratio, and PEG ratio
        JPanel volumePEMktCapPanel = new JPanel(new GridLayout(3, 1));

        // Labels for each metric; these will be updated with real values at runtime
        volumeLabel = new JLabel("Vol: ");
        peLabel = new JLabel("P/E: ");
        pegLabel = new JLabel("P/E/G: ");

        // Add these labels to the second panel (each gets a vertical slot)
        volumePEMktCapPanel.add(volumeLabel);
        volumePEMktCapPanel.add(peLabel);
        volumePEMktCapPanel.add(pegLabel);

        // ----- Third Column: 52W High, 52W Low, Market Cap -----    // This column displays the highest and lowest prices for the past year, and market capitalization
        JPanel rangeAndAvgVolPanel = new JPanel(new GridLayout(3, 1));

        // Labels for 52-week high, low, and market cap
        fiftyTwoWkHighLabel = new JLabel("52W H: ");
        fiftyTwoWkLowLabel = new JLabel("52W L: ");
        mktCapLabel = new JLabel("Mkt Cap: ");

        // Add each statistic to the third column panel
        rangeAndAvgVolPanel.add(fiftyTwoWkHighLabel);
        rangeAndAvgVolPanel.add(fiftyTwoWkLowLabel);
        rangeAndAvgVolPanel.add(mktCapLabel);

        // ----- Fourth Column: Percentage Change display -----    // This column displays the latest percentage change (e.g. after measuring two points on the chart)
        JPanel percentagePanel = new JPanel(new GridLayout(3, 1));

        // Label to show the current or last calculated percentage change
        percentageChange = new JLabel("Percentage Change");    // Only one label used, but the GridLayout makes all columns consistent
        percentagePanel.add(percentageChange);

        // ----- Add all columns to the 4-column data panel -----    // secondRowPanel should be a JPanel(new GridLayout(1, 4)) defined earlier
        secondRowPanel.add(openHighLowPanel);     // 1st column: Open/High/Low
        secondRowPanel.add(volumePEMktCapPanel);  // 2nd column: Volume, P/E, PEG
        secondRowPanel.add(rangeAndAvgVolPanel);  // 3rd column: 52W High/Low, Market Cap
        secondRowPanel.add(percentagePanel);      // 4th column: Percentage Change
        // ===== Summary =====    // This creates a summary bar with 4 vertical columns, each holding three (or one) labels.    // Layout makes data easy to read at a glance, and each label is updated dynamically elsewhere.

        // Add a border to the stock info section for clarity
        secondRowPanel.setBorder(BorderFactory.createTitledBorder("Stock Information"));

        // ========== FINAL ASSEMBLY ==========
        // Place the two major rows into the main chart tool panel
        mainPanel.add(firstRowPanel, BorderLayout.CENTER); // Top row: chart + news
        mainPanel.add(secondRowPanel, BorderLayout.SOUTH); // Bottom row: summary

        return mainPanel;
    }

    /**
     * Builds and returns a scroll pane containing the list of news items for the currently selected stock.
     * <p>
     * Supports single selection. Double-clicking a news item opens the full article in a dedicated dialog.
     *
     * @return JScrollPane containing the custom JList of News objects, ready to be placed in the GUI.
     */
    @NotNull
    private JScrollPane getNewsScrollPane() {
        // Create a JList for news articles using the model for this session
        JList<News> NewsList = new JList<>(NewsListModel);
        NewsList.setCellRenderer(new NotificationRenderer()); // Use your custom renderer
        NewsList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION); // Only allow one article selected at a time

        // Add a mouse listener for user interaction (clicks)
        NewsList.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                // If the user double-clicks (to open the news article)
                if (e.getClickCount() == 2) {
                    // Determine which item was clicked based on pointer position
                    int index = NewsList.locationToIndex(e.getPoint());
                    // Only proceed if a valid news item was clicked (not empty space)
                    if (index != -1) {
                        News clickedNews = NewsListModel.getElementAt(index);
                        openNews(clickedNews); // Show the news in a new dialog/window
                    }
                }
            }
        });

        // Return a scrollable container wrapping the news JList (ensures large news lists are always viewable)
        return new JScrollPane(NewsList);
    }

    /**
     * <h2>NotificationPanel</h2>
     * A compact Swing JPanel for displaying notifications.
     * Shows a colored strip, a title (with wrapping), and a clickable URL.
     * Used for lightweight pop-up alerts or in-app messages.
     */
    private static class NotificationPanel extends JPanel {

        /**
         * Constructs a notification panel with a colored strip, title, and clickable link.
         *
         * @param title the notification headline/title
         * @param url   the clickable link to display (e.g., "More info")
         * @param color the color for the vertical strip (e.g., sentiment or severity)
         */
        public NotificationPanel(String title, String url, Color color) {
            // --- Use BorderLayout with small horizontal and vertical gaps ---
            setLayout(new BorderLayout(8, 8));

            // --- Left: Color strip for visual emphasis (sentiment, status, etc.) ---
            JPanel colorStrip = new JPanel();
            colorStrip.setPreferredSize(new Dimension(4, 0)); // 4px wide, full height
            colorStrip.setBackground(color);                  // Set to supplied color
            add(colorStrip, BorderLayout.WEST);

            // --- Center: Content panel with vertical stacking ---
            JPanel contentPanel = new JPanel();
            contentPanel.setLayout(new BoxLayout(contentPanel, BoxLayout.Y_AXIS)); // Stack vertically
            contentPanel.setOpaque(false); // Transparent background (inherits parent color)

            // --- Title: Wrapped, bold, uneditable text area ---
            JTextArea titleLabel = new JTextArea(title);
            titleLabel.setLineWrap(true);                      // Enable line wrapping for long titles
            titleLabel.setWrapStyleWord(true);                 // Wrap at word boundaries
            titleLabel.setEditable(false);                     // Non-editable by user
            titleLabel.setFont(getFont().deriveFont(Font.BOLD)); // Bold font
            titleLabel.setOpaque(false);                       // Transparent background
            titleLabel.setAlignmentX(Component.LEFT_ALIGNMENT); // Align left in panel

            // --- URL label: clickable link displayed below the title ---
            JLabel urlLabel = new JLabel("<html><a href=''>" + url + "</a></html>");
            urlLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

            // --- Add components to the vertical stack (content panel) ---
            contentPanel.add(titleLabel);
            contentPanel.add(Box.createVerticalStrut(4)); // Small gap before URL
            contentPanel.add(urlLabel);

            // --- Add the content panel to the center of the NotificationPanel ---
            add(contentPanel, BorderLayout.CENTER);
        }
    }

    /**
     * Constructs the right-side 'Hype Panel' containing:
     * <ul>
     *   <li>A notification list for stock price alerts ("hype notifications")</li>
     *   <li>A log area for displaying system/user messages</li>
     *   <li>Section labels and flexible spacing for neat UI arrangement</li>
     * </ul>
     * <p>
     * This panel is designed to be placed on the right (EAST) of the main window.
     * It also sets up mouse listeners for user interaction and custom cell rendering for notifications.
     *
     * @return A ready-to-use JPanel, fully laid out and containing all subcomponents for notifications and logs.
     */
    public JPanel createHypePanel() {
        // The outermost vertical panel that holds everything in the hype section
        JPanel panel = new JPanel();
        // Set fixed width to avoid resizing when adding lots of notifications/logs
        panel.setPreferredSize(new Dimension(300, 0)); // (width, height)
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS)); // Stack children vertically

        // Top label for notification section
        JLabel notifications = new JLabel("Hype notifications");
        panel.add(notifications); // Add label at the top

        // === Notification list setup ===
        // (Re)initialize the model that holds the Notification objects for the list
        notificationListModel = new DefaultListModel<>();
        // Create the actual JList UI component to display notifications, backed by the above model
        notificationList = new JList<>(notificationListModel);
        notificationList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION); // Only one notification can be selected at a time

        // Attach a mouse listener for notification list interaction
        notificationList.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                // Optionally re-sort the notifications if shouldSort is enabled
                if (shouldSort) {
                    sortNotifications(globalByChange); // Sort by %change or by time, depending on user preference
                }
                // Double-click to open full notification details
                if (e.getClickCount() == 2) {
                    int index = notificationList.locationToIndex(e.getPoint());
                    if (index != -1) {
                        Notification clickedNotification = notificationListModel.getElementAt(index);
                        openNotification(clickedNotification); // Open the clicked notification in its window
                    }
                }
            }
        });

        // Use a custom cell renderer for notification appearance in the list
        notificationList.setCellRenderer((list, value, index, isSelected, cellHasFocus) -> {
            // Each notification is displayed as a colored label
            JLabel label = new JLabel(value.getTitle(), JLabel.CENTER); // Centered notification title

            // Set the background color for the notification (indicates type or importance)
            label.setOpaque(true); // Needed for background color to show up
            label.setBackground(value.getColor());

            // Give a nice thick black border with rounded corners
            label.setBorder(BorderFactory.createLineBorder(Color.BLACK, 2, true));
            // Set the text color based on selection (white for selected, black for normal)
            label.setForeground(isSelected ? Color.WHITE : Color.BLACK);

            return label; // This label is displayed for the notification cell
        });

        // Place the notification JList inside a scroll pane, for handling many alerts
        JScrollPane scrollPane = new JScrollPane(notificationList);
        scrollPane.setPreferredSize(new Dimension(200, 100)); // Reasonable height for short lists
        scrollPane.setMaximumSize(new Dimension(Integer.MAX_VALUE, Integer.MAX_VALUE)); // Allow it to grow tall if needed
        panel.add(scrollPane);

        // === Logging Area Setup ===
        // This area shows system, debug, or info messages to the user

        // Re-initialize the main log area as a text box (not editable by the user)
        logTextArea = new JTextArea(3, 20); // Initial size: 3 lines tall, 20 columns wide
        logTextArea.setEditable(false); // Prevent typing
        logTextArea.setLineWrap(true); // Auto-wrap long lines
        logTextArea.setWrapStyleWord(true); // Wrap on word boundaries for neatness

        // Place the log text area in a scroll pane so overflow can be reviewed
        JScrollPane logScrollPane = new JScrollPane(logTextArea);
        logScrollPane.setPreferredSize(new Dimension(200, 150)); // Reasonable log height
        // Restrict growth so log area doesn't take over the entire panel (when resizing window)
        logScrollPane.setMaximumSize(new Dimension(Integer.MAX_VALUE, 50));

        // Label for log section
        JLabel logLabel = new JLabel("Hype log window");
        panel.add(logLabel);       // Add label above log area
        panel.add(logScrollPane);  // Add log area below the label

        // === Spacing and Flexible Layout ===
        // Add a little space below the log area
        panel.add(Box.createRigidArea(new Dimension(0, 10))); // Fixed vertical gap
        // Add a flexible "glue" area to keep log at top, allowing notifications to grow downward
        panel.add(Box.createVerticalGlue()); // Pushes content up if there's extra space

        // (Redundant add, but safe: ensures log area is always included and padded)
        panel.add(logScrollPane);
        panel.add(Box.createRigidArea(new Dimension(0, 10))); // More space at bottom

        // Add a bold border/title to visually separate this panel in the main UI
        panel.setBorder(BorderFactory.createTitledBorder("Notifications"));

        // The finished right-side panel is returned to be added to the main UI frame
        return panel;
    }

    /**
     * Opens a Notification dialog. Ensures only one is visible at a time.
     * <p>
     * If another notification window is already open, closes it first, then shows the new one.
     *
     * @param notification The notification to display.
     */
    private void openNotification(Notification notification) {
        // Close any previous notification dialog (to avoid overlap/confusion)
        if (currentNotification != null) {
            currentNotification.closeNotification();
        }
        // Open/show the selected notification
        notification.showNotification();
        // Remember the new "current" notification as open, for tracking/closing later
        currentNotification = notification;
    }

    /**
     * Adds a news article to the global news list model.
     * <p>
     * Used for updating the company news section when new stories arrive from the API.
     * The news article is wrapped as a {@link News} object, which is then displayed in the
     * application's news list UI component.
     *
     * @param title              The headline or title of the news article.
     * @param content            The summary or excerpt of the news article.
     * @param url                The full URL to the news story (for opening in a browser or dialog).
     * @param sentimentForTicker The sentiment analysis result associated with the news article,
     *                           or {@code null} if not available. Used for sentiment coloring and badges.
     */
    public void addNews(String title, String content, String url, NewsResponse.TickerSentiment sentimentForTicker) {
        // Each news item is stored as a News object in the model,
        // which is rendered by the JList in the GUI.
        NewsListModel.addElement(new News(title, content, url, sentimentForTicker));
    }

    /**
     * Opens a News dialog, ensuring only one article is open at any time.
     * <p>
     * Closes the previously opened news dialog if one exists, then displays the new news article dialog.
     *
     * @param news The News object to display in detail.
     */
    // Open News and close previous one
    private void openNews(News news) {
        // If a news dialog is already open, close it before opening a new one
        if (CurrentNews != null) {
            CurrentNews.closeNews();
        }

        // Show the requested news article (opens its dialog or browser, depending on News implementation)
        news.showNews();

        // Set the newly opened news as the current one (so it can be closed next time)
        CurrentNews = news;
    }

    /**
     * Constructs the main application menu bar, including all top-level menus and their items,
     * as well as all event handlers for menu item clicks.
     * <p>
     * The menu bar includes:
     * <ul>
     *   <li>File: Config management (load/save/import/export/exit)</li>
     *   <li>Settings: Opens the settings dialog</li>
     *   <li>Hype Mode: Tools for detecting market rallies and enabling special analysis</li>
     *   <li>Notifications: Clear, sort (by change or date) notifications</li>
     * </ul>
     * <p>
     * All items are fully wired with event handlers, most of which are implemented as inner classes for clarity.
     *
     * @return The complete JMenuBar ready to be added to the main JFrame.
     */
    private JMenuBar createMenuBar() {
        // Main menu bar object (goes at the top of the main window)
        JMenuBar menuBar = new JMenuBar();

        // ===== Create the top-level menus (categories) =====
        JMenu file = new JMenu("File");               // For all config and file operations
        JMenu settings = new JMenu("Settings");        // For preferences/settings dialog
        JMenu hypeModeMenu = new JMenu("Hype mode");   // For special market tools/features
        JMenu Notifications = new JMenu("Notifications"); // For manipulating the notification list

        // ===== File Menu Items (config and app control) =====
        JMenuItem load = new JMenuItem("Load the config (manually again)"); // Reload config from disk
        JMenuItem importC = new JMenuItem("Import config");                 // Load config from external file
        JMenuItem exportC = new JMenuItem("Export config");                 // Save current config to file
        JMenuItem save = new JMenuItem("Save the config");                  // Save config to disk
        JMenuItem exit = new JMenuItem("Exit (saves)");                     // Exit application (with save)

        // ===== Settings Menu Item =====
        JMenuItem settingHandler = new JMenuItem("Open settings"); // Opens the GUI for app settings

        // ===== Hype Mode Menu Items =====
        JMenuItem checkForRallies = new JMenuItem("Check market for current Rally's"); // Detect market rallies
        JMenuItem activateHypeMode = new JMenuItem("Activate hype mode");              // Start hype mode (algorithmic scan)

        // ===== Notifications Menu Items =====
        JMenuItem clear = new JMenuItem("Clear Notifications");                   // Remove all notifications from UI
        JMenuItem sortChange = new JMenuItem("Sort Notifications by Change");     // Sort by % change (largest first)
        JMenuItem sortDate = new JMenuItem("Sort Notifications by Date");         // Sort by time (most recent first)

        // ===== Assemble File Menu =====
        file.add(load);
        file.add(importC);
        file.add(exportC);
        file.add(save);
        file.add(exit);

        // ===== Assemble Settings Menu =====
        settings.add(settingHandler);

        // ===== Assemble Hype Mode Menu =====
        hypeModeMenu.add(checkForRallies);
        hypeModeMenu.add(activateHypeMode);

        // ===== Assemble Notifications Menu =====
        Notifications.add(clear);
        Notifications.add(sortChange);
        Notifications.add(sortDate);

        // ===== Add all menus to the main menu bar =====
        menuBar.add(file);
        menuBar.add(settings);
        menuBar.add(hypeModeMenu);
        menuBar.add(Notifications);

        // ===== Wire up all event handlers =====
        // Config management
        load.addActionListener(new eventLoad());         // Custom inner class: handles loading config
        save.addActionListener(new eventSave());         // Handles saving current config
        exit.addActionListener(new eventExit());         // Handles saving & exiting
        importC.addActionListener(new eventImport());    // Handles importing config from file
        exportC.addActionListener(new eventExport());    // Handles exporting config to file

        // Settings dialog
        settingHandler.addActionListener(new eventSettings()); // Open settings dialog

        // Hype tools
        checkForRallies.addActionListener(new checkForRallies());          // Start market rally scan
        activateHypeMode.addActionListener(new eventActivateHypeMode());   // Start hype mode

        // Notification controls (sorting, clearing)
        clear.addActionListener(e -> notificationListModel.clear()); // Lambda for quick clear
        sortChange.addActionListener(new eventSortNotifications(true));  // Sort by %change
        sortDate.addActionListener(new eventSortNotifications(false));   // Sort by date

        // ===== Done! Return the fully assembled and wired menu bar =====
        return menuBar;
    }

    /**
     * Sorts all notifications in the notification list, either by percent change or by timestamp (date).
     * <p>
     * This method updates the global sorting preference, converts the {@link DefaultListModel}
     * to a standard List for sorting, sorts in place, then clears and repopulates the model.
     *
     * @param byChange If true, sorts by percentage change (descending); if false, sorts by date (most recent first).
     */
    public void sortNotifications(boolean byChange) {
        // Remember user sorting preference (used by mouse listener for auto-sorting)
        globalByChange = byChange;

        // Convert the notification list model (which is not sortable) to a mutable List
        List<Notification> notifications = Collections.list(notificationListModel.elements());

        if (byChange) {
            // Sort by the .getChange() property in descending order (largest changes first)
            notifications.sort(Comparator.comparingDouble(Notification::getChange).reversed());
        } else {
            // Sort by date (most recent notifications first)
            notifications.sort(Comparator.comparing(Notification::getLocalDateTime).reversed());
        }

        // Clear the original model and re-add notifications in new order
        notificationListModel.clear();
        notificationListModel.addAll(notifications);
    }

    /**
     * Simple private helper class for storing aggregated OHLC stock data for a specific time period.
     * Used internally for candlestick chart construction.
     * <p>
     * Each instance stores open, high, low, and close for a particular time window (minute, hour, day, etc.).
     */
    private static class AggregatedStockData {
        double open, high, low, close;
    }

    /**
     * Event handler class for exporting the application configuration to a user-specified file location.
     * Implements {@link ActionListener} so it can be wired directly to a menu or button.
     */
    public static class eventExport implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            // Locate the existing config.xml file in the project root
            Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
            File configFile = configPath.toFile();

            // Prevent export if the config doesn't exist (show error dialog)
            if (!configFile.exists()) {
                JOptionPane.showMessageDialog(null, "config.xml not found in the project root!");
                return;
            }

            // Prompt the user with a file chooser for export location
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setDialogTitle("Export Configuration");

            // Open the dialog and capture the result (user click)
            int userSelection = fileChooser.showSaveDialog(null);

            // Only proceed if the user approved the save action
            if (userSelection == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile();

                try {
                    // Copy the config.xml file to the chosen location (overwrite if exists)
                    Files.copy(configFile.toPath(), selectedFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    JOptionPane.showMessageDialog(null, "Configuration exported successfully!");
                } catch (IOException ex) {
                    // Show error if file copy fails
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(null, "Error exporting configuration file: " + ex.getMessage());
                }
            }
        }
    }

    /**
     * Event handler for activating 'Hype Mode', which performs special market scanning/analysis.
     * This action runs on a background thread to avoid blocking the UI.
     * <p>
     * It invokes {@code mainDataHandler.startHypeMode(volume)}.
     * Any exceptions are printed to standard error.
     */
    public static class eventActivateHypeMode implements ActionListener {
        // Single-threaded executor ensures only one Hype Mode scan at a time
        private static final ExecutorService executorService = Executors.newSingleThreadExecutor();

        @Override
        public void actionPerformed(ActionEvent e) {
            // Run the heavy analysis in the background so UI stays responsive
            executorService.submit(() -> {
                try {
                    // Kick off the mainDataHandler's hype mode analysis
                    mainDataHandler.startHypeMode(market, volume);
                } catch (Exception ex) {
                    // Log any error (for debugging)
                    ex.printStackTrace();
                }
            });
        }
    }

    /**
     * Opens the application settings/configuration dialog when triggered.
     * <p>
     * Shows a non-blocking settings window, pre-populated with current values.
     */
    public static class eventSettings implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // Create settings GUI dialog using current config values
            settingsHandler gui = new settingsHandler(volume, symbols, shouldSort, apiKey, useRealtime,
                    aggressiveness, useCandles, t212ApiToken, pushCutUrlEndpoint, greed, market);
            gui.setSize(500, 700);            // Fixed dialog size
            gui.setAlwaysOnTop(true);         // Ensures settings stays above main window
            gui.setTitle("Config handler ");  // Dialog window title
            gui.setVisible(true);             // Show the settings window
        }
    }

    /**
     * Event handler for saving the current configuration state.
     * <p>
     * Calls {@link mainUI#saveConfig(String[][])} with all current values.
     */
    public static class eventSave implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            saveConfig(getValues()); // Save all current settings to persistent storage
        }
    }

    /**
     * Event handler for exiting the application.
     * <p>
     * Saves configuration before exiting to ensure no user changes are lost.
     */
    public static class eventExit implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            saveConfig(getValues());          // Save config as safety
            System.out.println("Exit application");
            System.exit(0);                   // Terminate application
        }
    }

    /**
     * Event handler for importing a configuration file.
     * <p>
     * Presents file chooser, overwrites the app config, reloads UI, and confirms with a dialog.
     */
    public static class eventImport implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            // Prompt user to select a file to import
            JFileChooser fileChooser = new JFileChooser();
            fileChooser.setDialogTitle("Import Configuration");

            int userSelection = fileChooser.showOpenDialog(null); // Show dialog

            // If user picked a file, process import
            if (userSelection == JFileChooser.APPROVE_OPTION) {
                File selectedFile = fileChooser.getSelectedFile();

                // Destination is config.xml in the current working directory
                Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
                File configFile = configPath.toFile();

                try {
                    // Overwrite old config.xml with new one
                    Files.copy(selectedFile.toPath(), configFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

                    loadConfig(); // Reload everything (UI, data, API keys, etc)

                    JOptionPane.showMessageDialog(null, "Configuration imported and saved as config.xml successfully!");

                } catch (IOException ex) {
                    ex.printStackTrace();
                    JOptionPane.showMessageDialog(null, "Error processing configuration file: " + ex.getMessage());
                }
            }
        }
    }

    /**
     * Event handler for manually reloading the configuration file from disk.
     * <p>
     * Useful if user manually edits config or imports a file externally.
     */
    public static class eventLoad implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            loadConfig(); // Load settings and refresh the application state
        }
    }

    /**
     * Simple data record storing a ticker symbol and maximum open quantity allowed for that security.
     * <p>
     * Used to store metadata about each tradable instrument.
     */
    public record TickerData(String ticker, int maxOpenQuantity) {
    }

    /**
     * ActionListener implementation to trigger a scan for stocks currently in a "rally" (strong upward movement).
     * <p>
     * This class ensures the expensive backend computation (API/network scan) runs on a background thread,
     * while the GUI update (showing results) always occurs on the Event Dispatch Thread for Swing safety.
     */
    public class checkForRallies implements ActionListener {
        // Single-thread executor: ensures only one rally scan runs at a time, and keeps the UI responsive
        private static final ExecutorService executorService = Executors.newSingleThreadExecutor();

        @Override
        public void actionPerformed(ActionEvent e) {
            // When the menu/button is clicked, immediately start the scan in the background.
            executorService.submit(() -> {
                try {
                    // Request the list of all stocks currently detected as "in rally" from the backend handler.
                    // This could be a slow API call, so it must NOT block the Swing UI.
                    List<String> rallyStocks = mainDataHandler.checkForRallies();

                    // Once the scan is complete, switch back to the Event Dispatch Thread to update the UI.
                    // SwingUtilities.invokeLater guarantees Swing thread-safety for all GUI changes.
                    SwingUtilities.invokeLater(() -> showRallyWindow(rallyStocks));

                } catch (Exception ex) {
                    // Catch any errors in the background thread to avoid silent failures.
                    ex.printStackTrace();
                }
            });
        }

        /**
         * Display a window listing all rally candidate stocks, each with actions to view or add to watchlist.
         * This dialog is always constructed and shown on the Swing Event Dispatch Thread.
         *
         * @param rallyStocks The list of symbols identified by the backend as rallying.
         */
        private void showRallyWindow(List<String> rallyStocks) {
            // Create a new dialog window for results, not modal so user can continue using the app.
            JFrame frame = new JFrame("Rally Candidates");
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

            // Set the size large enough for results but don't exceed screen height.
            frame.setSize(500, Math.min(600, rallyStocks.size() * 60 + 100));
            frame.setLocationRelativeTo(null); // Center the window on the screen for user convenience

            // Vertical box layout for stacking candidate rows vertically
            JPanel content = new JPanel();
            content.setLayout(new BoxLayout(content, BoxLayout.Y_AXIS));
            content.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10)); // Adds uniform padding

            // For every stock found, create a UI row: symbol label, "View" and "Add to Watchlist" buttons
            for (String symbol : rallyStocks) {
                JPanel row = new JPanel(new FlowLayout(FlowLayout.LEFT)); // Horizontal layout for this row

                JLabel label = new JLabel(symbol);              // Stock symbol label (e.g., "AAPL")
                label.setPreferredSize(new Dimension(100, 25)); // Force consistent width for all symbols

                // View button: sets this stock as the selected stock in the main UI and refreshes chart/news
                JButton viewButton = new JButton("View Stock");
                viewButton.addActionListener(ev -> handleStockSelection(symbol)); // Lambda: on click, focus chart

                // Add to Watchlist button: leverages mainUI utility, updates watchlist for user
                JButton watchlistButton = mainUI.getJButton(symbol, frame);

                // Assemble row: label, view, add buttons (horizontal)
                row.add(label);
                row.add(viewButton);
                row.add(watchlistButton);
                content.add(row); // Stack this row vertically in the results panel
            }

            // If the result list is long, make it scrollable
            JScrollPane scroll = new JScrollPane(content);
            scroll.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);

            frame.setContentPane(scroll);  // Set the scrollable panel as the window's main content
            frame.setVisible(true);        // Finally, show the dialog
        }
    }

    /**
     * Handler for sorting notifications either by % change or by most recent date.
     * <p>
     * Constructed with the sort type, and can be wired to a menu item directly.
     */
    public class eventSortNotifications implements ActionListener {
        boolean byChange;

        /**
         * @param change If true, sort by percent change; else, sort by recency/date.
         */
        public eventSortNotifications(boolean change) {
            this.byChange = change;
        }

        @Override
        public void actionPerformed(ActionEvent e) {
            sortNotifications(byChange);
        }
    }

    /**
     * <h2>NotificationRenderer</h2>
     * Custom ListCellRenderer for displaying {@link News} objects in a JList
     * as stylized notification cards, using NotificationPanel for layout.
     * Handles theme colors, selection highlighting, and sentiment-based borders.
     */
    public static class NotificationRenderer implements ListCellRenderer<News> {

        /**
         * Configures and returns the component used to render each News item in the list.
         * Applies selection colors, foreground/background, and sentiment border styling.
         *
         * @param list         the JList we're painting
         * @param news         the News object for this row
         * @param index        the index of the cell in the list
         * @param isSelected   true if the cell is selected
         * @param cellHasFocus true if the cell has focus
         * @return the component to display for the given list cell
         */
        @Override
        public Component getListCellRendererComponent(JList<? extends News> list, News news, int index,
                                                      boolean isSelected, boolean cellHasFocus) {
            // Create a notification card panel with title, url, and sentiment color
            NotificationPanel panel = new NotificationPanel(
                    news.getTitle(),
                    news.getUrl(),
                    news.getSentimentColor()
            );

            // Set background and foreground based on selection state and theme
            Color bg = UIManager.getColor(isSelected ? "List.selectionBackground" : "List.background");
            Color fg = UIManager.getColor(isSelected ? "List.selectionForeground" : "List.foreground");

            panel.setBackground(bg);
            panel.setForeground(fg);

            // By default, show a matte border with the sentiment color
            panel.setBorder(BorderFactory.createMatteBorder(0, 3, 1, 0, news.getSentimentColor()));

            // If selected, overlay a thicker border for focus/selection
            if (isSelected) {
                panel.setBorder(BorderFactory.createCompoundBorder(
                        BorderFactory.createLineBorder(UIManager.getColor("List.selectionForeground"), 2),
                        BorderFactory.createMatteBorder(0, 3, 1, 0, news.getSentimentColor())
                ));
            }

            return panel;
        }
    }
}