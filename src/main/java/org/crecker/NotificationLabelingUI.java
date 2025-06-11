package org.crecker;

import com.crazzyghost.alphavantage.timeseries.response.StockUnit;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.CandlestickRenderer;
import org.jfree.chart.ui.Layer;
import org.jfree.data.xy.DefaultOHLCDataset;
import org.jfree.data.xy.OHLCDataItem;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.crecker.pLTester.dumpNotifications;

/**
 * A Swing-based UI for manually labeling notifications as “Good” or “Bad”
 * by displaying a candlestick chart for each notification’s historical and future stock windows.
 */
public class NotificationLabelingUI extends JFrame {

    /**
     * Ordered list of notifications to be labeled by the user.
     */
    private final List<Notification> notifications;

    /**
     * Index of the current notification being displayed.
     */
    private int currentIndex = 0;

    // UI components
    /**
     * Label showing the progress counter (e.g., "1 / 100").
     */
    private final JLabel counterLabel = new JLabel();

    /**
     * Panel that displays the price or data chart for the current notification.
     */
    private final ChartPanel chartPanel;

    /**
     * Button to mark a notification as "Good".
     */
    private final JButton goodButton;

    /**
     * Button to skip labeling the current notification.
     */
    private final JButton skipButton;

    /**
     * Button to mark a notification as "Bad".
     */
    private final JButton badButton;

    /**
     * Constructor: Initializes the UI with the given list of notifications.
     * Sets up the window, chart area, and vertically stacked action buttons.
     *
     * @param notifications Ordered list of {@link Notification} objects to label.
     */
    public NotificationLabelingUI(List<Notification> notifications) {
        super("Notification Labeling Suite");                 // Set window title
        this.notifications = notifications;                      // Store provided notifications

        // Apply system look-and-feel for native appearance
        try {
            UIManager.setLookAndFeel(
                    UIManager.getSystemLookAndFeelClassName()
            );
            SwingUtilities.updateComponentTreeUI(this);
        } catch (Exception ignored) {
            // If look-and-feel setup fails, continue with default
        }

        // Configure main frame behavior and layout
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);         // Exit application on close
        setLayout(new BorderLayout(8, 8));                      // Layout with 8px gaps
        setPreferredSize(new Dimension(1000, 700));             // Initial window size

        // --- Top region: progress counter ---
        counterLabel.setFont(
                counterLabel.getFont().deriveFont(Font.BOLD, 16f)
        );                                                       // Bold, larger text
        counterLabel.setBorder(
                BorderFactory.createEmptyBorder(10, 0, 10, 0)
        );                                                       // Padding around label
        counterLabel.setHorizontalAlignment(JLabel.CENTER);      // Center text
        add(counterLabel, BorderLayout.NORTH);

        // --- Center region: chart container ---
        chartPanel = new ChartPanel(null);                       // Placeholder chart
        JPanel chartContainer = new JPanel(new BorderLayout());
        chartContainer.setBorder(
                BorderFactory.createTitledBorder("Price Chart")
        );                                                       // Title around chart area
        chartContainer.add(chartPanel, BorderLayout.CENTER);
        add(chartContainer, BorderLayout.CENTER);

        // --- Side region: vertical button stack ---
        JPanel buttonPanel = new JPanel();                       // Panel to hold buttons vertically
        buttonPanel.setLayout(
                new BoxLayout(buttonPanel, BoxLayout.Y_AXIS)
        );                                                       // Vertical stacking
        buttonPanel.setBorder(
                BorderFactory.createEmptyBorder(20, 20, 20, 20)
        );                                                       // Padding around buttons

        // Initialize action buttons with oversized dimensions
        goodButton = new JButton("Good");                       // Good label
        styleButton(goodButton, new Color(0x00AA00));            // Dark green background
        goodButton.setMaximumSize(new Dimension(150, 60));       // Force larger size
        goodButton.setAlignmentX(Component.CENTER_ALIGNMENT);     // Center in panel

        skipButton = new JButton("Skip");                       // Skip label
        styleButton(skipButton, new Color(0x888888));            // Gray background
        skipButton.setMaximumSize(new Dimension(150, 60));       // Force larger size
        skipButton.setAlignmentX(Component.CENTER_ALIGNMENT);     // Center in panel

        badButton = new JButton("Bad");                         // Bad label
        styleButton(badButton, new Color(0xCC0000));             // Dark red background
        badButton.setMaximumSize(new Dimension(150, 60));       // Force larger size
        badButton.setAlignmentX(Component.CENTER_ALIGNMENT);     // Center in panel

        // Add buttons with spacing between them
        buttonPanel.add(goodButton);
        buttonPanel.add(Box.createRigidArea(new Dimension(0, 10)));
        buttonPanel.add(skipButton);
        buttonPanel.add(Box.createRigidArea(new Dimension(0, 10)));
        buttonPanel.add(badButton);

        // Add the button panel to the east side for easy reach
        add(buttonPanel, BorderLayout.EAST);

        // --- Wire up button actions to label or skip ---
        goodButton.addActionListener(
                e -> labelCurrentNotification(1)
        );                                                        // 1 = Good
        badButton.addActionListener(
                e -> labelCurrentNotification(0)
        );                                                        // 0 = Bad
        skipButton.addActionListener(
                e -> skipCurrentNotification()
        );                                                        // Skip

        // Finalize frame layout and show first notification
        pack();                                                   // Arrange components
        setLocationRelativeTo(null);                              // Center window on screen
        updateUIForCurrent();                                     // Display the first notification
    }

    /**
     * Applies consistent styling to toolbar buttons.
     * Sets font, background, foreground, and border properties for a uniform look.
     *
     * @param button The {@link JButton} to style.
     * @param bg     Background {@link Color} for the button.
     */
    private void styleButton(JButton button, Color bg) {
        button.setFont(new Font("SansSerif", Font.BOLD, 18));  // Larger, bold text
        button.setBackground(bg);                                // Custom background color
        button.setForeground(Color.WHITE);                       // White text for contrast
        button.setOpaque(true);                                  // Paint background
        button.setBorderPainted(false);                          // Remove default border
    }

    /**
     * Advances past the current notification without labeling it.
     */
    private void skipCurrentNotification() {
        currentIndex++;
        updateUIForCurrent();
    }

    /**
     * Updates the counter label and the chartPanel for the notification at {@code currentIndex}.
     * <p>
     * If all notifications have been labeled (currentIndex >= size), disables buttons and shows a “finished” message.
     * </p>
     */
    private void updateUIForCurrent() {
        // If we’ve reached or passed the end of the list:
        if (currentIndex >= notifications.size()) {
            // Display finished message and disable UI elements
            counterLabel.setText("All notifications labeled!"); // Show final text
            chartPanel.setChart(null);                           // Clear any existing chart
            goodButton.setEnabled(false);                        // Disable Good button
            skipButton.setEnabled(false);
            badButton.setEnabled(false);                         // Disable Bad button
            return;                                               // Done
        }

        // Otherwise, get the current notification to display:
        Notification notif = notifications.get(currentIndex);

        // Update counter text to “Notification X of N”
        counterLabel.setText(String.format(
                "Notification %d of %d",
                currentIndex + 1,           // 1-based index for user
                notifications.size()        // Total count
        ));

        // Create a candlestick chart for this notification using historical and future windows
        JFreeChart chart = createCandlestickChart(
                notif.getStockUnitList(),           // Historical list of StockUnit
                notif.getValidationWindow()         // Future‐window list (up to 10 bars)
        );
        chartPanel.setChart(chart);             // Set chart into the panel for display
    }

    /**
     * Called when the user clicks “Good” or “Bad” to label the current Notification.
     * <p>
     * Sets the current notification’s target, writes it to disk (via dumpNotifications),
     * and advances to the next notification.
     * </p>
     *
     * @param targetValue 1 indicates “Good”, 0 indicates “Bad”.
     */
    private void labelCurrentNotification(int targetValue) {
        // Retrieve the current notification object
        Notification notif = notifications.get(currentIndex);
        // Assign the user’s label to the notification
        notif.setTarget(targetValue);

        try {
            // Append this single notification to the persistent store (e.g., CSV or JSON file)
            dumpNotifications(new ArrayList<>(Collections.singleton(notif)));
        } catch (IOException ex) {
            ex.printStackTrace();
            // Note: We choose to continue even if writing fails, but could show an error dialog
        }

        // Advance to the next notification index
        currentIndex++;
        // Refresh UI for the newly current notification (or end state)
        updateUIForCurrent();
    }

    /**
     * Creates a candlestick chart with two distinct background regions:
     * <ul>
     *     <li>Historical region (stockUnits) shaded light gray</li>
     *     <li>Future region (validationWindow) shaded pale yellow</li>
     * </ul>
     *
     * @param historical   The list of {@link StockUnit} representing the past window (chronological).
     * @param futureWindow The list of {@link StockUnit} representing the next up to 10 bars (future period).
     * @return A {@link JFreeChart} instance showing the combined OHLC data.
     */
    private JFreeChart createCandlestickChart(
            List<StockUnit> historical,
            List<StockUnit> futureWindow
    ) {
        // 1) Convert both historical and futureStock lists into a single array of OHLCDataItem
        int totalBars = historical.size() + futureWindow.size();   // Total bars to plot
        OHLCDataItem[] dataItems = new OHLCDataItem[totalBars];    // Allocate array

        // Populate dataItems with historical bars first
        for (int i = 0; i < historical.size(); i++) {
            StockUnit u = historical.get(i);                   // Get ith historical StockUnit
            double open = u.getOpen();                         // Opening price
            double high = u.getHigh();                         // Highest price
            double low = u.getLow();                           // Lowest price
            double close = u.getClose();                       // Closing price
            double volume = u.getVolume();                     // Traded volume
            // Convert the LocalDateTime in StockUnit to java.util.Date (date-only precision)
            dataItems[i] = new OHLCDataItem(
                    u.getDateDate(), open, high, low, close, volume
            );
        }
        // Populate dataItems with future bars immediately following the historical ones
        for (int j = 0; j < futureWindow.size(); j++) {
            StockUnit u = futureWindow.get(j);                 // Get jth future StockUnit
            double open = u.getOpen();                         // Opening price
            double high = u.getHigh();                         // Highest price
            double low = u.getLow();                           // Lowest price
            double close = u.getClose();                       // Closing price
            double volume = u.getVolume();                     // Traded volume
            dataItems[historical.size() + j] = new OHLCDataItem(
                    u.getDateDate(), open, high, low, close, volume
            );
        }

        // 2) Build a DefaultOHLCDataset from our dataItems
        String seriesKey = "Price Series";                      // Label for the dataset
        DefaultOHLCDataset dataset = new DefaultOHLCDataset(seriesKey, dataItems);

        // 3) Create the candlestick chart (no legend, null title)
        JFreeChart chart = ChartFactory.createCandlestickChart(
                null,                // No chart title
                "Date",              // X-axis label
                "Price",             // Y-axis label
                dataset,             // The OHLC dataset we just created
                false                // Do not show a legend
        );

        // 4) Customize the plot: add colored background intervals
        XYPlot plot = (XYPlot) chart.getPlot();                // Extract plot from chart
        plot.setBackgroundPaint(Color.WHITE);                  // Set plain white background

        // Configure Y-axis (price axis) to auto-range, excluding zero if needed
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);                          // Auto-scale the range
        rangeAxis.setAutoRangeIncludesZero(false);             // Don’t force zero in range

        // a) Mark historical region with light gray
        // Determine start and end Dates for historical region
        java.util.Date histStart = dataItems[0].getDate();                              // First bar’s date
        java.util.Date histEnd = dataItems[historical.size() - 1].getDate();           // Last historical bar’s date
        IntervalMarker histMarker = new IntervalMarker(
                histStart.getTime(),      // Lower bound in ms since epoch
                histEnd.getTime()         // Upper bound in ms since epoch
        );
        histMarker.setPaint(new Color(0x777777));   // Light gray paint
        histMarker.setAlpha(0.5f);                  // 50% opacity
        plot.addDomainMarker(histMarker, Layer.BACKGROUND); // Add behind the candlesticks

        // b) Mark future region with pale yellow, only if futureWindow is non-empty
        if (!futureWindow.isEmpty()) {
            java.util.Date futStart = dataItems[historical.size()].getDate();          // First future bar’s date
            java.util.Date futEnd = dataItems[totalBars - 1].getDate();                // Last bar’s date
            IntervalMarker futMarker = new IntervalMarker(
                    futStart.getTime(),    // Lower bound for future window
                    futEnd.getTime()       // Upper bound for future window
            );
            futMarker.setPaint(new Color(0xFFFACD));  // Pale yellow (“LemonChiffon”)
            futMarker.setAlpha(0.5f);                 // 50% opacity
            plot.addDomainMarker(futMarker, Layer.BACKGROUND); // Add behind the candlesticks
        }

        // 5) Tweak date‐axis formatting for X-axis
        DateAxis domainAxis = (DateAxis) plot.getDomainAxis(); // Get the date axis
        domainAxis.setAutoRange(true);                          // Auto-scale date range
        domainAxis.setLowerMargin(0.01);                        // Small left margin so bars don’t butt against edge
        domainAxis.setUpperMargin(0.01);                        // Small right margin

        // 6) Adjust candlestick renderer properties
        CandlestickRenderer renderer = (CandlestickRenderer) plot.getRenderer();
        renderer.setAutoWidthMethod(CandlestickRenderer.WIDTHMETHOD_SMALLEST);
        // Automatically compute bar width as small as possible based on data spacing

        // Return the fully constructed and styled chart
        return chart;
    }
}