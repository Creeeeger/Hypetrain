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

import static org.crecker.mainUI.getTicker;
import static org.crecker.mainUI.useCandles;

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

    // Main notification data and charting fields
    private final String title;                 // The notification's display title
    private final String content;               // Main textual content for this notification
    private final List<StockUnit> stockUnitList;// Stock data (price/time) relevant to this event
    private final LocalDateTime localDateTime;  // Date/time for the notification event
    private final String symbol;                // Stock ticker symbol
    private final double change;                // Percent change in price that triggered this event
    private final int config;                   // Config code: event type (used for color)
    private final Color color;                  // Notification highlight color (from config)
    private final TimeSeries timeSeries;        // For line charts
    private final OHLCSeries ohlcSeries;        // For candlestick charts

    JLabel percentageChange;                    // Label for showing percentage difference between two points
    private JFrame notificationFrame;           // Frame/window displaying this notification
    private ChartPanel chartPanel;              // Chart panel for the graph (JFreeChart)

    /**
     * Create a Notification instance for a specific event, including relevant stock data and type.
     *
     * @param title         Title of the notification window.
     * @param content       Description or explanation of the event.
     * @param stockUnitList List of StockUnit objects (price/time).
     * @param localDateTime Time of the event.
     * @param symbol        Stock symbol this notification refers to.
     * @param change        Percentage change for the notification (e.g. dip/spike).
     * @param config        Event type (0=dip, 1=gap, 2=R-line spike, 3=spike, else=purple).
     */
    public Notification(String title, String content, List<StockUnit> stockUnitList, LocalDateTime localDateTime, String symbol, double change, int config) {
        this.title = title;
        this.content = content;
        this.stockUnitList = stockUnitList;
        this.localDateTime = localDateTime;
        this.symbol = symbol;
        this.change = change;
        this.config = config;

        /*
          config 0 dip         - bright red (major drop)
          config 1 gap filler  - deep orange
          config 2 R-line spike- blue
          config 3 spike       - green
          else                 - royal purple
         */
        if (config == 0) {
            this.color = new Color(255, 0, 0);         // Bright Red
        } else if (config == 1) {
            this.color = new Color(255, 140, 0);       // Deep Orange
        } else if (config == 2) {
            this.color = new Color(0, 128, 255);       // Sky Blue
        } else if (config == 3) {
            this.color = new Color(34, 177, 76);       // Leaf Green
        } else {
            this.color = new Color(128, 0, 128);       // Royal Purple
        }

        // Build series for chart plotting
        this.ohlcSeries = new OHLCSeries(symbol + " OHLC");
        processOHLCData(stockUnitList); // Populate OHLC for candlestick

        this.timeSeries = new TimeSeries(symbol + " Price");
        processTimeSeriesData(stockUnitList); // Populate for line chart
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
    private static XYPlot getXyPlot(JFreeChart chart) {
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
                    new Second(unit.getDateDate()), // X: time (second-level precision)
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
            Second period = new Second(unit.getDateDate());
            // Add to candlestick series
            ohlcSeries.add(new OHLCItem(
                    period,
                    unit.getOpen(),
                    unit.getHigh(),
                    unit.getLow(),
                    unit.getClose()
            ));
            // Add (or update) in time series (for line chart)
            timeSeries.addOrUpdate(period, unit.getClose());
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
        chartPanel = createChart();
        chartPanel.setPreferredSize(new Dimension(600, 320)); // Wider chart

        // Bottom panel for buttons and labels
        JPanel bottomPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));

        percentageChange = new JLabel("Percentage Change"); // Placeholder label for user analysis

        // In-app "superchart" button: opens real-time chart for this symbol
        JButton openRealTime = new JButton("Open in Realtime SuperChart");

        openRealTime.addActionListener(e -> mainUI.getInstance().handleStockSelection(this.symbol));

        // Button: open web portal for external charting (e.g. Trading212)
        String tickerCode = getTicker(symbol);
        String url = "https://app.trading212.com/?lang=de&ticker=" + tickerCode;

        JButton openWebPortal = new JButton("Open in Web Portal");
        openWebPortal.setForeground(Color.BLUE);
        openWebPortal.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        openWebPortal.setFocusPainted(false);
        openWebPortal.setBorderPainted(false);
        openWebPortal.setContentAreaFilled(false);

        openWebPortal.addActionListener(e -> {
            try {
                Desktop.getDesktop().browse(new URI(url));
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        });

        // Add buttons and label to the bottom panel
        bottomPanel.add(openWebPortal);
        bottomPanel.add(percentageChange);
        bottomPanel.add(openRealTime);

        // Assemble the panels into the main notification window
        mainPanel.add(scrollPane, BorderLayout.NORTH);
        mainPanel.add(chartPanel, BorderLayout.CENTER);
        mainPanel.add(bottomPanel, BorderLayout.SOUTH);

        // Add assembled panel to the frame
        notificationFrame.add(mainPanel);
        notificationFrame.validate();
        notificationFrame.repaint();

        // Display the notification window
        notificationFrame.setVisible(true);
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