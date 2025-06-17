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
import org.jfree.data.time.Minute;
import org.jfree.data.time.Second;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;
import org.jfree.data.time.ohlc.OHLCItem;
import org.jfree.data.time.ohlc.OHLCSeries;
import org.jfree.data.time.ohlc.OHLCSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.net.URI;
import java.time.LocalDateTime;
import java.util.List;

import static org.crecker.mainUI.*;

/**
 * Notification represents an event popup for a specific stock.
 * It contains event info, charting, and interactivity features for analysis.
 * Handles drawing of markers and shaded regions, and provides links to further details.
 */
public class Notification {
    // Static state for cross-notification marker placement (shared for all instances)
    private static double point1X = Double.NaN;
    private static double point1Y = Double.NaN;
    private static double point2X = Double.NaN;
    private static double point2Y = Double.NaN;
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;
    private String notificationId;

    // Main notification data and charting fields
    private String title;                       // The notification's display title
    private final String content;               // Main textual content for this notification
    private final List<StockUnit> stockUnitList;// Stock data (price/time) relevant to this event
    private List<StockUnit> validationWindow;   // relevant for ML training
    private List<StockUnit> normalizedList;     // lookback window after min-max normalization (prices & volume scaled to [0,1]) for charting and inference
    private final LocalDateTime localDateTime;  // Date/time for the notification event
    private final String symbol;                // Stock ticker symbol
    private final double change;                // Percent change in price that triggered this event
    private final int config;                   // Config code: event type (used for color)
    private final Color color;                  // Notification highlight color (from config)
    private final TimeSeries timeSeries;        // For line charts
    private final OHLCSeries ohlcSeries;        // For candlestick charts
    private int target = 0;                     // ML label

    JLabel percentageChange;                    // Label for showing percentage difference between two points
    private JFrame notificationFrame;           // Frame/window displaying this notification
    private final ChartPanel chartPanel;        // Chart panel for the graph (JFreeChart)

    /**
     * Create a Notification instance for a specific event, including relevant stock data and type.
     *
     * @param title            Title of the notification window.
     * @param content          Description or explanation of the event.
     * @param stockUnitList    List of StockUnit objects (price/time).
     * @param localDateTime    Time of the event.
     * @param symbol           Stock symbol this notification refers to.
     * @param change           Percentage change for the notification (e.g., dip/spike).
     * @param config           Event type code, which determines the visual highlight color:
     *                         <ul>
     *                           <li>1 – Gap filler (deep orange)</li>
     *                           <li>2 – R-line spike (blue)</li>
     *                           <li>3 – Spike (green)</li>
     *                           <li>4 – Uptrend (royal purple)</li>
     *                           <li>5 – Second-based alarm (alarm red)</li>
     *                           <li>Other – Gray</li>
     *                         </ul>
     *                         <p>Use bright colors to enhance text visibility.</p>
     * @param validationWindow A list of subsequent bars immediately following this event,
     *                         used as the ML training/labeling window. It must contain at least
     *                         {@code frameSize} entries. The first {@code frameSize} bars of this
     *                         list are examined to determine whether the initial signal was
     *                         “good” (e.g., price moved above a threshold) or “bad” (e.g., price
     *                         failed to follow through). If fewer than {@code frameSize} bars are
     *                         available, the notification is skipped or labeled as invalid.
     */
    public Notification(String title, String content, List<StockUnit> stockUnitList, LocalDateTime localDateTime, String symbol, double change, int config, List<StockUnit> validationWindow) {
        this.title = title;
        this.content = content;
        this.stockUnitList = stockUnitList;
        this.localDateTime = localDateTime;
        this.symbol = symbol;
        this.change = change;
        this.config = config;
        this.validationWindow = validationWindow;
        this.normalizedList = null;

        /*
          config 1 gap filler   - deep orange
          config 2 R-line spike - blue
          config 3 spike        - green
          config 4 uptrend      - royal purple
          config 5 second based - Alarm red
          other                 - gray
          -- use brighter colors in order to see text better and more clear --
         */

        if (config == 1) {
            this.color = new Color(255, 171, 70);       // Deep Orange
        } else if (config == 2) {
            this.color = new Color(48, 149, 255);       // Sky Blue
        } else if (config == 3) {
            this.color = new Color(60, 184, 93);       // Leaf Green
        } else if (config == 4) {
            this.color = new Color(255, 58, 255);       // Royal Purple
        } else if (config == 5) {
            this.color = new Color(255, 0, 0, 255);   // alarm red
        } else {
            this.color = new Color(147, 147, 159);        // Gray
        }

        // Build series for chart plotting
        this.ohlcSeries = new OHLCSeries(symbol + " OHLC");
        processOHLCData(stockUnitList); // Populate OHLC for candlestick

        this.timeSeries = new TimeSeries(symbol + " Price");
        processTimeSeriesData(stockUnitList); // Populate for line chart

        // initialize chart at the beginning
        chartPanel = createChart();
    }

    /**
     * Sets the list of normalized StockUnit instances for this notification.
     * <p>
     * This list typically represents the lookback window after min-max normalization,
     * with price and volume values scaled into [0,1].
     * </p>
     *
     * @param normalizedList a non-null List of {@link StockUnit} objects whose fields
     *                       (open, high, low, close, volume) have been normalized.
     */
    public void setNormalizedList(List<StockUnit> normalizedList) {
        this.normalizedList = normalizedList;
    }

    /**
     * Retrieves the normalized StockUnit list for this notification.
     * <p>
     * This list contains the lookback window data after min-max scaling,
     * ready for charting or further processing.
     * </p>
     *
     * @return a List of {@link StockUnit} instances with normalized fields,
     * or null if no normalization has been applied yet.
     */
    public List<StockUnit> getNormalizedList() {
        return normalizedList;
    }

    /**
     * Returns the notification ID.
     *
     * @return the notification ID as a String
     */
    public String getNotificationId() {
        return notificationId;
    }

    /**
     * Sets the notification ID.
     *
     * @param id the notification ID to set
     */
    public void setNotificationId(String id) {
        this.notificationId = id;
    }

    /**
     * Sets the ML label (target).
     *
     * @param target the integer class label to assign (e.g., 0 for negative, 1 for positive)
     */
    public void setTarget(int target) {
        this.target = target;
    }

    /**
     * Returns the ML label (target).
     *
     * @return the integer class label currently stored in this instance
     */
    public int getTarget() {
        return target;
    }

    /**
     * Updates the title text of this notification.
     *
     * @param title New title string to assign.
     */
    public void setTitle(String title) {
        this.title = title;
    }

    /**
     * Sets the validation window for this stock to the specified list of {@link StockUnit} objects.
     * <p>
     * This replaces any existing data in the validation window. The provided list should contain
     * all StockUnit entries relevant to the desired validation period.
     *
     * @param validationWindow the list of {@link StockUnit} objects to set as the validation window; must not be null
     */
    public void setValidationWindow(List<StockUnit> validationWindow) {
        this.validationWindow = validationWindow;
    }

    /**
     * Returns the list of {@link StockUnit} objects representing the validation window for this stock.
     * <p>
     * The validation window contains all StockUnit entries within the relevant validation period,
     * typically used for model validation, performance checks, or backtesting.
     *
     * @return a list of {@link StockUnit} objects in the validation window
     */
    public List<StockUnit> getValidationWindow() {
        return validationWindow;
    }

    /**
     * Adds the first vertical marker (for user analysis) and removes any previous markers/shading.
     *
     * @param plot      XYPlot to which the marker will be added.
     * @param xPosition X-axis (time) value for the marker.
     */
    private static void addFirstMarker(XYPlot plot, double xPosition) {
        // Remove any previous markers or shaded regions before placing new marker
        if (marker1 != null) {
            plot.removeDomainMarker(marker1);
        }
        if (marker2 != null) {
            plot.removeDomainMarker(marker2);
        }
        if (shadedRegion != null) {
            plot.removeDomainMarker(shadedRegion);
        }

        // Create the marker (vertical line) and add it to the plot
        marker1 = new ValueMarker(xPosition);
        marker1.setPaint(Color.GREEN);  // Marker is green
        marker1.setStroke(new BasicStroke(2.5f));  // Marker thickness
        plot.addDomainMarker(marker1);
    }

    /**
     * Configures and returns the XYPlot for candlestick chart display.
     *
     * @param chart JFreeChart to extract the plot from.
     * @return XYPlot with candlestick renderer set.
     */
    @NotNull
    public static XYPlot getXyPlot(JFreeChart chart) {
        XYPlot plot = chart.getXYPlot();

        // Set up custom candlestick renderer for visual clarity
        CandlestickRenderer renderer = new CandlestickRenderer();
        renderer.setAutoWidthMethod(CandlestickRenderer.WIDTHMETHOD_SMALLEST);
        renderer.setUpPaint(Color.GREEN);    // Up candle (close >= open)
        renderer.setDownPaint(Color.RED);    // Down candle (close < open)
        renderer.setUseOutlinePaint(true);   // Draw candle outline
        renderer.setDrawVolume(false);       // Do not show volume bars
        plot.setRenderer(renderer);
        return plot;
    }

    // ===================== Accessor Methods =====================

    /**
     * Gets the highlight color used for this notification (based on config type).
     *
     * @return Color for notification/event type.
     */
    public Color getColor() {
        return color;
    }

    /**
     * Gets the price change (as a percentage) associated with this notification.
     *
     * @return The percentage change that triggered the notification.
     */
    public double getChange() {
        return change;
    }

    /**
     * Gets the ticker symbol of the stock relevant to this notification.
     *
     * @return Stock ticker (e.g., "AAPL", "MSFT").
     */
    public String getSymbol() {
        return symbol;
    }

    /**
     * Gets the timestamp for the event or data point that triggered this notification.
     *
     * @return LocalDateTime of the event.
     */
    public LocalDateTime getLocalDateTime() {
        return localDateTime;
    }

    /**
     * Gets the title text for this notification popup.
     *
     * @return Notification title.
     */
    public String getTitle() {
        return title;
    }

    /**
     * Gets the detailed content/description for this notification.
     *
     * @return The full textual content.
     */
    public String getContent() {
        return content;
    }

    /**
     * Returns the list of StockUnit objects (OHLC/time data) associated with this notification.
     *
     * @return List of StockUnit price records.
     */
    public List<StockUnit> getStockUnitList() {
        return stockUnitList;
    }

    /**
     * Returns the configuration integer (type code) for this notification.
     *
     * @return Config value indicating the type of event (used for color, etc).
     */
    public int getConfig() {
        return config;
    }

    /**
     * Gets the current chart panel component used for displaying the event's graph.
     *
     * @return The ChartPanel showing the JFreeChart.
     */
    public ChartPanel getChartPanel() {
        return chartPanel;
    }

    /**
     * Gets the time series (close prices) for use in line chart mode.
     *
     * @return TimeSeries containing close price vs. time.
     */
    public TimeSeries getTimeSeries() {
        return timeSeries;
    }

    /**
     * Gets the OHLC series (open/high/low/close data) for use in candlestick chart mode.
     *
     * @return OHLCSeries containing full candle data.
     */
    public OHLCSeries getOHLCSeries() {
        return ohlcSeries;
    }

    // ===================== Chart Data Construction/Mutation Methods =====================

    /**
     * Populates the OHLCSeries with all items from a list of StockUnit records.
     * Each StockUnit is mapped to an OHLCItem (time, open, high, low, close).
     *
     * @param stockUnits List of StockUnit objects to convert.
     */
    private void processOHLCData(List<StockUnit> stockUnits) {
        for (StockUnit unit : stockUnits) {
            ohlcSeries.add(new OHLCItem(
                    new Minute(unit.getDateDate()), // X: time (minute-level precision)
                    unit.getOpen(),                 // Y: open price
                    unit.getHigh(),                 // Y: high price
                    unit.getLow(),                  // Y: low price
                    unit.getClose()                 // Y: close price
            ));
        }
    }

    /**
     * Populates the TimeSeries with all close price data from a list of StockUnit records.
     *
     * @param stockUnits List of StockUnit objects to convert.
     */
    private void processTimeSeriesData(List<StockUnit> stockUnits) {
        for (StockUnit unit : stockUnits) {
            Second period = new Second(unit.getDateDate()); // Use exact second for time axis
            timeSeries.add(period, unit.getClose());
        }
    }

    /**
     * Adds a new data point to both the OHLC and TimeSeries if it is newer than the event time.
     * Triggers a UI update if data is added.
     *
     * @param unit New StockUnit record to add.
     */
    public void addDataPoint(StockUnit unit) {
        // Only add data points that are after the notification event's timestamp
        if (unit.getLocalDateTimeDate().isAfter(this.localDateTime)) {
            Minute period = new Minute(unit.getDateDate());
            // Add to candlestick series
            ohlcSeries.add(new OHLCItem(
                    period,
                    unit.getOpen(),
                    unit.getHigh(),
                    unit.getLow(),
                    unit.getClose()
            ));
            try {
                // Add (or update) in time series (for line chart)
                timeSeries.addOrUpdate(period, unit.getClose());
            } catch (Exception e) {
                timeSeries.addOrUpdate(new Second(unit.getDateDate()), unit.getClose());
            }

            // Refresh chart UI after data update
            updateUI();
        }
    }

    /**
     * Triggers a repaint of the chart on the Swing event thread.
     * Ensures thread-safety for UI updates.
     */
    private void updateUI() {
        SwingUtilities.invokeLater(() -> chartPanel.repaint());
    }

    /**
     * Shows the notification window, assembling all UI and chart elements.
     */
    public void showNotification() {
        // Create the notification window/frame
        notificationFrame = new JFrame(title);
        notificationFrame.setSize(700, 500); // Increased window size for better chart
        notificationFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        notificationFrame.setLocationRelativeTo(null); // Center window on screen
        notificationFrame.setAlwaysOnTop(true);        // Always display on top

        // Create the text area for event description, with line wrapping enabled
        JTextArea textArea = new JTextArea(content);
        textArea.setEditable(false);
        textArea.setLineWrap(true);
        textArea.setWrapStyleWord(true);

        // Place text area in a scroll pane
        JScrollPane scrollPane = new JScrollPane(textArea);

        // Main container panel with border layout for UI arrangement
        JPanel mainPanel = new JPanel(new BorderLayout());

        // Chart panel: line or candlestick chart, based on user setting
        chartPanel.setPreferredSize(new Dimension(600, 320)); // Chart area size

        // Bottom panel for buttons and labels
        JPanel bottomPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        percentageChange = new JLabel("Percentage Change"); // Info label

        // --- Buttons ---

        // Create a button that opens a detailed real-time chart within the app
        JButton openRealTime = new JButton("Open in Realtime SuperChart");
        // Attach an action: when clicked, trigger the mainUI to show the chart for this symbol
        openRealTime.addActionListener(e -> mainUI.getInstance().handleStockSelection(this.symbol));

        // Prepare a button that links to an external trading platform (e.g. Trading212) for this symbol
        String tickerCode = getTicker(symbol);  // Convert to web-usable ticker
        String url = "https://app.trading212.com/?lang=de&ticker=" + tickerCode;

        JButton openWebPortal = new JButton("Open in Web Portal");

        // Change mouse cursor to a hand pointer for better UX (like a web link)
        openWebPortal.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        // Remove button focus outline (looks cleaner in dark mode)
        openWebPortal.setFocusPainted(false);

        // Define the action: open the Trading212 URL in the default browser when clicked
        openWebPortal.addActionListener(e -> {
            try {
                Desktop.getDesktop().browse(new URI(url));
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        });

        JButton showRulesBtn = getShowRulesBtn();

        // === DARK MODE STYLING ===

        // Define core dark mode colors for consistent appearance
        Color panelBg = new Color(35, 35, 40);     // Background for panels and text areas
        Color textFg = new Color(230, 230, 230);  // Foreground color for text and labels
        Color buttonBg = new Color(45, 45, 60);     // Button background
        Color buttonFg = new Color(220, 220, 220);  // Button foreground/text
        Color accentBlue = new Color(86, 156, 214);   // Highlight blue for special buttons

        // === Apply dark styling to the main layout and components ===

        // Set dark background for the main content panel
        mainPanel.setBackground(panelBg);
        // Set the scroll pane's viewport (where the text is) to dark
        scrollPane.getViewport().setBackground(panelBg);
        // Text area inherits the dark background and light foreground
        textArea.setBackground(panelBg);
        textArea.setForeground(textFg);

        // Set the background for the bottom bar (where buttons and labels sit)
        bottomPanel.setBackground(panelBg);
        // Make the percentageChange label text light for contrast
        percentageChange.setForeground(textFg);

        // --- Unified Button Styling ---
        // Bundle buttons into an array for easy styling in a loop
        JButton[] buttons = {openWebPortal, openRealTime, showRulesBtn};
        // Each button gets its border color (blue for web, white for realtime)
        Color[] buttonBorders = {accentBlue, buttonFg, buttonFg};
        // Each button gets its text color (blue for web, white for realtime)
        Color[] buttonTexts = {accentBlue, buttonFg, buttonFg};

        // Loop through each button and apply all style settings for consistency
        for (int i = 0; i < buttons.length; i++) {
            JButton btn = buttons[i];
            btn.setBackground(buttonBg);                    // Use the dark background
            btn.setForeground(buttonTexts[i]);              // Text color (blue or white)
            btn.setOpaque(true);                            // Ensure background is painted
            btn.setBorder(BorderFactory.createLineBorder(buttonBorders[i])); // Colored border
            btn.setFocusPainted(false);                     // Remove focus outline for a cleaner look
        }

        // --- Add all controls to their panels in order ---

        // Add buttons and the info label to the bottom panel (order: left to right)
        bottomPanel.add(openWebPortal);
        bottomPanel.add(percentageChange);
        bottomPanel.add(openRealTime);
        bottomPanel.add(showRulesBtn);

        // Place each section in the main window layout
        mainPanel.add(scrollPane, BorderLayout.NORTH);   // Text at the top
        mainPanel.add(chartPanel, BorderLayout.CENTER);  // Chart in the middle
        mainPanel.add(bottomPanel, BorderLayout.SOUTH);  // Controls at the bottom

        // === Final Assembly and Display ===

        // Add the assembled main panel to the JFrame
        notificationFrame.add(mainPanel);
        // Ensure the frame layout and component painting is up to date
        notificationFrame.validate();
        notificationFrame.repaint();
        // Make the window visible to the user
        notificationFrame.setVisible(true);
    }

    /**
     * Creates and returns a JButton configured to display a dialog with trading rules
     * in a visually styled, dark-themed HTML format. When clicked, the button opens
     * a modal dialog showing the trading rules with clearly color-coded sections for
     * entries, exits, warnings, and meta/mental models.
     *
     * @return a configured JButton labeled "Show Trading Rules", which opens the styled rules dialog on click.
     *
     * <p><b>Usage:</b> Attach the returned button to your Swing GUI. Requires a valid {@code notificationFrame} as the dialog parent.
     */
    @NotNull
    private JButton getShowRulesBtn() {
        // Create a new JButton with a descriptive label
        JButton showRulesBtn = new JButton("Show Trading Rules");

        // Attach an action listener to respond to button clicks
        showRulesBtn.addActionListener(e -> {
            // ------- THEME OVERRIDES: Apply dark mode settings to JOptionPane dialogs --------
            // This ensures any JOptionPane (including the rules dialog) uses a dark background and light text.
            UIManager.put("Panel.background", new Color(35, 35, 40));
            UIManager.put("OptionPane.background", new Color(35, 35, 40));
            UIManager.put("OptionPane.messageForeground", new Color(230, 230, 230));

            // ------- HTML CONTENT: Define trading rules with color coding and structure -------
            // The rules are encoded in HTML for structured display.
            // Color references:
            //   - #61ef75 (green): Entry rules
            //   - #ff4747 (red): Exit rules and "never" instructions
            //   - #ffa900 (orange): Warnings
            //   - #0acbf4 (cyan): Meta/mental model emphasis
            String rulesHtml =
                    "<html><body style='background-color:#232328; color:#f0f0f0; font-family:monospace; font-size:13pt;'>" +
                            // Title
                            "<div style='margin-bottom:6px;'><span style='color:#ff4747; font-size:15pt;'>🛑 TRADING EXECUTION — LIVE DISCIPLINE</span></div>" +
                            "<div style='color:#d0d0d0; margin-bottom:9px;'>Stay tactical. <b>Obey the rules</b>, protect the account.</div>" +
                            // Entry rules
                            "<div><span style='color:#61ef75; font-weight:bold;'>Entry Rules:</span></div>" +
                            "<ul style='margin-top:3px; margin-bottom:10px;'>" +
                            "<li><b><span style='color:#61ef75;'>ONLY</span> enter</b> when the <span style='color:#61ef75;'>next candle closes higher</span> than the previous. <br>" +
                            "<span style='color:#ffa900;'>(Avoid catching a falling knife; minimize entry risk.)</span></li>" +
                            "<li><b><span style='color:#ff4747;'>NEVER</span> buy into a red candle</b> just because it looks cheap.</li>" +
                            "</ul>" +
                            // Exit rules
                            "<div><span style='color:#ff4747; font-weight:bold;'>Exit Rules:</span></div>" +
                            "<ul style='margin-top:3px; margin-bottom:10px;'>" +
                            "<li><b><span style='color:#ff4747;'>EXIT</span> immediately</b> if the trend momentum <b>flattens out</b>—don’t wait or hope.</li>" +
                            "<li><b>Sell</b> after the <span style='color:#ff4747;'>first sharp drop</span>, ideally <b>during</b> the drop—not after. <br>" +
                            "<span style='color:#ffa900;'>(Momentum dies fast. No mercy, no bag-holding.)</span></li>" +
                            "</ul>" +
                            // Mental model/meta rules
                            "<div><span style='color:#0acbf4; font-weight:bold;'>Mental Model:</span></div>" +
                            "<ul style='margin-top:3px;'>" +
                            "<li>This system does <span style='color:#ff4747;'>NOT</span> chase every pump; only <b>clean, high-momentum entries</b>.</li>" +
                            "<li><span style='color:#ff4747;'>Cut losers fast.</span> Goal: stay in the game, not be a hero.</li>" +
                            "<li>If the setup isn’t <b>perfect</b>, <span style='color:#ff4747;'>do NOT trade</span>. Wait for the <span style='color:#0acbf4;'>A+ scenario</span>.</li>" +
                            "</ul>" +
                            "</body></html>";

            // ------- JEditorPane for HTML display --------
            // JEditorPane renders the rules as styled HTML. Not editable for safety.
            JEditorPane rulesPane = new JEditorPane("text/html", rulesHtml);
            rulesPane.setEditable(false); // Prevents any user modification
            rulesPane.setBackground(new Color(35, 35, 40)); // Consistent with dialog
            rulesPane.setForeground(new Color(230, 230, 230)); // Readable text
            rulesPane.setCaretPosition(0); // Scroll to top by default

            // ------- SCROLL PANE: Ensure dialog is large and scrollable --------
            // Allows the user to see all content even on small screens.
            JScrollPane rulesScroll = new JScrollPane(rulesPane);
            rulesScroll.setPreferredSize(new Dimension(680, 350)); // Wide and tall for easy reading

            // ------- SHOW DIALOG: Display the trading rules in a modal dialog -------
            // Uses notificationFrame as the parent for modality and placement.
            JOptionPane.showMessageDialog(
                    notificationFrame,
                    rulesScroll,
                    "Trading Rules",
                    JOptionPane.INFORMATION_MESSAGE
            );
        });

        // Return the fully configured button for use in the UI
        return showRulesBtn;
    }

    /**
     * Creates and returns the appropriate chart panel (candlestick or line) based on global setting.
     *
     * @return ChartPanel for UI display.
     */
    private ChartPanel createChart() {
        // Decide which chart to display based on the current visualization mode.
        if (useCandles) {
            // Show candlestick chart if enabled.
            return createOHLCChart();
        } else {
            // Otherwise, show the standard time series (line) chart.
            return createTimeSeriesChart();
        }
    }

    /**
     * Creates and configures a line chart panel displaying the time series data (close price vs. time).
     * Supports interactive marker and shaded region placement by mouse click for user analysis.
     *
     * @return ChartPanel for line chart visualization.
     */
    private ChartPanel createTimeSeriesChart() {
        // Prepare a dataset with the TimeSeries (close prices over time)
        TimeSeriesCollection dataset = new TimeSeriesCollection(timeSeries);

        // Create the JFreeChart object for time series visualization
        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                symbol + " Price Chart", // Chart title
                "Date",                  // X-axis label
                "Price",                 // Y-axis label
                dataset,                 // Dataset
                true,                    // Show legend
                true,                    // Use tooltips
                false                    // No URLs
        );

        // enable chart dark mode
        setDarkMode(chart);

        // Customize plot appearance and range
        XYPlot plot = chart.getXYPlot();
        plot.getDomainAxis().setAutoRange(true);    // Automatically scale X-axis
        plot.getRangeAxis().setAutoRange(true);     // Automatically scale Y-axis
        // Add a faint green background region to emphasize chart area
        plot.addRangeMarker(new IntervalMarker(0.0, Double.POSITIVE_INFINITY, new Color(200, 255, 200, 100)));

        // Create the panel that will hold and render the chart
        ChartPanel chartPanel = new ChartPanel(chart);

        // Add interactivity: marker and region placement by mouse click
        chartPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                // Translate the clicked screen coordinates to chart's data coordinates
                Point2D p = chartPanel.translateScreenToJava2D(e.getPoint());
                ValueAxis xAxis = plot.getDomainAxis();
                ValueAxis yAxis = plot.getRangeAxis();

                // Convert mouse position to data coordinates
                double x = xAxis.java2DToValue(p.getX(), chartPanel.getScreenDataArea(), plot.getDomainAxisEdge());
                double y = yAxis.java2DToValue(p.getY(), chartPanel.getScreenDataArea(), plot.getRangeAxisEdge());

                if (Double.isNaN(point1X)) {
                    // First click: place the first marker (vertical line)
                    point1X = x;
                    point1Y = y;
                    addFirstMarker(plot, point1X);
                } else {
                    // Second click: place the second marker and shaded region between two points
                    point2X = x;
                    point2Y = y;
                    addSecondMarkerAndShade(plot);

                    // Reset static state for next marker placement
                    point1X = Double.NaN;
                    point1Y = Double.NaN;
                    point2X = Double.NaN;
                    point2Y = Double.NaN;
                }
            }
        });

        return chartPanel;
    }

    /**
     * Creates and configures a candlestick chart panel (OHLC data visualization).
     * Supports the same marker/shaded region interactivity as the line chart.
     *
     * @return ChartPanel for candlestick chart visualization.
     */
    private ChartPanel createOHLCChart() {
        // Prepare the OHLC data series as a dataset for the candlestick chart
        OHLCSeriesCollection dataset = new OHLCSeriesCollection();
        dataset.addSeries(ohlcSeries);

        // Create the candlestick chart using JFreeChart's factory
        JFreeChart chart = ChartFactory.createCandlestickChart(
                symbol + " OHLC Chart", // Chart title
                "Date",                 // X-axis label
                "Price",                // Y-axis label
                dataset,                // Dataset
                true                    // Show legend
        );

        // enable chart dark mode
        setDarkMode(chart);

        // Get and configure the plot for custom rendering
        XYPlot plot = getXyPlot(chart);

        // Set axes to auto-range and exclude zero from Y-axis to prevent distortion
        plot.getDomainAxis().setAutoRange(true);
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);
        rangeAxis.setAutoRangeIncludesZero(false); // Don't force Y-axis to include zero
        rangeAxis.configure(); // Apply changes immediately

        // Add a faint green region in the background
        plot.addRangeMarker(new IntervalMarker(0.0, Double.POSITIVE_INFINITY, new Color(200, 255, 200, 100)));

        // Create the panel that will hold and render the chart
        ChartPanel chartPanel = new ChartPanel(chart);

        // Add marker and region interactivity (same logic as line chart)
        chartPanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Point2D p = chartPanel.translateScreenToJava2D(e.getPoint());
                ValueAxis xAxis = plot.getDomainAxis();
                ValueAxis yAxis = plot.getRangeAxis();

                // Convert mouse position to data coordinates
                double x = xAxis.java2DToValue(p.getX(), chartPanel.getScreenDataArea(), plot.getDomainAxisEdge());
                double y = yAxis.java2DToValue(p.getY(), chartPanel.getScreenDataArea(), plot.getRangeAxisEdge());

                if (Double.isNaN(point1X)) {
                    // First click: place the first marker (vertical line)
                    point1X = x;
                    point1Y = y;
                    addFirstMarker(plot, point1X);
                } else {
                    // Second click: place the second marker and shaded region between two points
                    point2X = x;
                    point2Y = y;
                    addSecondMarkerAndShade(plot);

                    // Reset static state for next marker placement
                    point1X = Double.NaN;
                    point1Y = Double.NaN;
                    point2X = Double.NaN;
                    point2Y = Double.NaN;
                }
            }
        });

        return chartPanel;
    }

    /**
     * Places the second marker (vertical line) and adds a shaded region between the two selected X positions.
     * Computes and updates the percentage change between the two selected Y values.
     *
     * @param plot The plot to which markers/regions are added.
     */
    private void addSecondMarkerAndShade(XYPlot plot) {
        // Remove previous second marker and shaded region if they exist
        if (marker2 != null) {
            plot.removeDomainMarker(marker2);
        }
        if (shadedRegion != null) {
            plot.removeDomainMarker(shadedRegion);
        }

        // Calculate percentage change between the two selected Y values
        double percentageDiff = ((point2Y - point1Y) / point1Y) * 100;

        // Select color for the marker and shaded region: green for positive, red for negative
        Color markerColor = (percentageDiff >= 0) ? Color.GREEN : Color.RED;
        Color shadeColor = (percentageDiff >= 0) ? new Color(100, 200, 100, 50) : new Color(200, 100, 100, 50);

        // Place the second marker at the selected X position
        marker2 = new ValueMarker(point2X);
        marker2.setPaint(markerColor);
        marker2.setStroke(new BasicStroke(1.5f)); // Customize thickness
        plot.addDomainMarker(marker2);

        // Add a shaded region between the two selected X points (translucent fill)
        shadedRegion = new IntervalMarker(Math.min(point1X, point2X), Math.max(point1X, point2X));
        shadedRegion.setPaint(shadeColor);  // Translucent green or red shade
        plot.addDomainMarker(shadedRegion);

        // Update the label in the UI with the calculated percentage difference
        percentageChange.setText(String.format("Percentage Change: %.3f%%", percentageDiff));
    }

    /**
     * Closes and disposes the notification window if open.
     */
    public void closeNotification() {
        if (notificationFrame != null) {
            notificationFrame.dispose();
        }
    }

    /**
     * String representation for notification (shown in notification lists).
     *
     * @return Title of the notification.
     */
    @Override
    public String toString() {
        return title; // Display the title in the list
    }
}