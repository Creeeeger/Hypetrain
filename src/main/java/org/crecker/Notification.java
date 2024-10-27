package org.crecker;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;

import javax.swing.*;
import java.awt.*;

public class Notification {
    private final String title;
    private final String content;
    private final TimeSeries timeSeries;
    private JFrame notificationFrame; // Frame for the notification

    public Notification(String title, String content, TimeSeries timeSeries) {
        this.title = title;
        this.content = content;
        this.timeSeries = timeSeries;
    }

    public String getTitle() { //get the title
        return title;
    }

    public String getContent() { //get the content
        return content;
    }

    public TimeSeries getTimeSeries() { //get the data chart (timeSeries)
        return timeSeries;
    }

    public JFrame getNotificationFrame() { //get the frame I don't know why but still then we got it
        return notificationFrame;
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

        // Add the scroll pane with the text area to the top of the panel
        mainPanel.add(scrollPane, BorderLayout.NORTH);

        // Wrap the TimeSeries in a TimeSeriesCollection, which implements XYDataset
        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(timeSeries);

        // Create the chart with the dataset
        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                "View window of change", // Chart title
                "Date", // X-axis Label
                "Price", // Y-axis Label
                dataset, // The dataset (now a TimeSeriesCollection)
                true, // Show legend
                true, // Show tooltips
                false // Show URLs
        );

        // Enable zoom and pan features on the chart panel
        ChartPanel chartPanel = new ChartPanel(chart);

        // Add the chart panel to the main panel
        mainPanel.add(chartPanel, BorderLayout.CENTER);

        // Add the main panel to the notification window
        notificationFrame.add(mainPanel);

        // Make the notification visible
        notificationFrame.setVisible(true);
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
}