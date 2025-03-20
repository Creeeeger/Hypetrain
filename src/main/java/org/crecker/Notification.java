package org.crecker;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.time.LocalDateTime;

public class Notification {
    private final String title;
    private final String content;
    private final TimeSeries timeSeries;
    private final LocalDateTime localDateTime;
    private final String symbol;
    private final double change;
    private JFrame notificationFrame; // Frame for the notification
    private final boolean dip;
    private final Color color;
    JLabel percentageChange;
    private ChartPanel chartPanel;

    private static double point1X = Double.NaN;
    private static double point1Y = Double.NaN;
    private static double point2X = Double.NaN;
    private static double point2Y = Double.NaN;
    private static ValueMarker marker1 = null;
    private static ValueMarker marker2 = null;
    private static IntervalMarker shadedRegion = null;

    public Notification(String title, String content, TimeSeries timeSeries, LocalDateTime localDateTime, String symbol, double change, boolean dip) {
        this.title = title;
        this.content = content;
        this.timeSeries = timeSeries;
        this.localDateTime = localDateTime;
        this.symbol = symbol;
        this.change = change;
        this.dip = dip;
        if (dip) {
            this.color = new Color(250, 30, 13);

        } else {
            this.color = new Color(33, 215, 13);
        }
    }

    public Color getColor() {
        return color;
    }

    public double getChange() {
        return change;
    }

    public String getSymbol() {
        return symbol;
    }

    public LocalDateTime getLocalDateTime() {
        return localDateTime;
    }

    public String getTitle() {
        return title;
    }

    public String getContent() { //get the content
        return content;
    }

    public TimeSeries getTimeSeries() { //get the data chart (timeSeries)
        return timeSeries;
    }

    public boolean isDip() {
        return dip;
    }

    public void showNotification() {
        // Create the notification window
        notificationFrame = new JFrame(title);
        notificationFrame.setSize(600, 400); // Adjust the size for both text and chart
        notificationFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        notificationFrame.setLocationRelativeTo(null);
        notificationFrame.setAlwaysOnTop(true);

        // Create the text area with content, enable line wrapping
        JTextArea textArea = new JTextArea(content);
        textArea.setEditable(false);
        textArea.setLineWrap(true);         // Enable line wrapping
        textArea.setWrapStyleWord(true);    // Wrap at word boundaries

        // Add the text area to a scroll pane
        JScrollPane scrollPane = new JScrollPane(textArea);

        // Create the panel to hold both text and chart
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout());

        // Enable zoom and pan features on the chart panel
        chartPanel = createChart(timeSeries, symbol + " price Chart");
        chartPanel.setPreferredSize(new Dimension(500, 300));

        // Create a panel to hold the percentage label and button
        JPanel bottomPanel = new JPanel(new FlowLayout(FlowLayout.CENTER)); // Centers elements
        percentageChange = new JLabel("Percentage Change");
        JButton openRealTime = new JButton("Open in Realtime SuperChart");

        openRealTime.addActionListener(e -> Main_UI.getInstance().handleStockSelection(this.symbol));

        bottomPanel.add(percentageChange);
        bottomPanel.add(openRealTime);

        mainPanel.add(scrollPane, BorderLayout.NORTH);
        mainPanel.add(chartPanel, BorderLayout.CENTER);
        mainPanel.add(bottomPanel, BorderLayout.SOUTH); // The bottom panel now holds both elements

        // Add the main panel to the notification window
        notificationFrame.add(mainPanel);
        notificationFrame.validate();
        notificationFrame.repaint();

        // Make the notification visible
        notificationFrame.setVisible(true);
    }

    private ChartPanel createChart(TimeSeries timeSeries, String chartName) {
        // Wrap the TimeSeries in a TimeSeriesCollection
        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(timeSeries);

        // Create the chart with the dataset
        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                chartName, // Chart title
                "Date",    // X-axis Label
                "Price",   // Y-axis Label
                dataset,   // The dataset
                true,      // Show legend
                true,      // Show tooltips
                false      // Show URLs
        );

        // Customize the plot
        XYPlot plot = chart.getXYPlot();
        plot.addRangeMarker(new IntervalMarker(0.0, Double.POSITIVE_INFINITY, new Color(200, 255, 200, 100)));

        // Create the chart panel and enable zoom and pan features
        ChartPanel chartPanel = new ChartPanel(chart);

        // Add mouse listener for marker placement and shaded region
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

        return chartPanel;
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

    private void addSecondMarkerAndShade(XYPlot plot) {
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

    public void closeNotification() {
        if (notificationFrame != null) {
            notificationFrame.dispose();
        }
    }

    @Override
    public String toString() {
        return title; // Display the title in the list
    }

    public ChartPanel getChartPanel() {
        return chartPanel;
    }
}