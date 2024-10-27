package org.crecker;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.ValueMarker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.time.Second;
import org.jfree.data.time.TimeSeries;
import org.jfree.data.time.TimeSeriesCollection;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.*;

public class Main_UI extends JFrame {
    static int vol;
    static float hyp;
    static boolean isSorted;
    static String sym;
    static String selected_stock = "Select a stock"; //selected_stock is the stock to show in the chart bar
    static String[][] setting_data;
    static JPanel symbol_panel, chart_tool_panel, hype_panel, chartPanel;
    static JMenuBar menuBar;
    static JMenu file, settings, hype_mode_menu, Notifications;
    static JMenuItem load, save, exit, setting_handler, activate_hype_mode, clear, sort;
    static JTextField searchField;
    static JLabel openLabel, highLabel, lowLabel, volumeLabel, peLabel, mktCapLabel, fiftyTwoWkHighLabel, fiftyTwoWkLowLabel, avgVolumeLabel;
    static JButton removeButton, addButton, oneDayButton, threeDaysButton, oneWeekButton, twoWeeksButton, oneMonthButton, threeMonthsButton, sixMonthsButton, oneYearButton;
    static DefaultListModel<String> stockListModel;
    static Map<String, Color> stockColors;
    static ChartPanel chartDisplay; // This will hold the chart
    static TimeSeries timeSeries; // Your time series data

    static DefaultListModel<Notification> notificationListModel;
    static JList<Notification> notificationList;
    static Notification currentNotification; // Track currently opened notification

    static DefaultListModel<News> NewsListModel;
    static JList<News> NewsList;
    static News CurrentNews;


    public Main_UI() {
        // Setting layout for the frame (1 row, 4 columns)
        setLayout(new BorderLayout());
        BorderFactory.createTitledBorder("Stock monitor");

        // Menu bar
        setJMenuBar(createMenuBar());

        //Panels
        symbol_panel = create_symbol_panel();
        chart_tool_panel = create_chart_tool_panel();
        hype_panel = create_hype_panel();

        // Add sections to the frame using BorderLayout
        add(symbol_panel, BorderLayout.WEST);
        add(chart_tool_panel, BorderLayout.CENTER);
        add(hype_panel, BorderLayout.EAST);
    }

    public static void main(String[] args) {
        Main_UI gui = new Main_UI();
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        gui.setSize(1900, 1000); // Width and height of the window
        gui.setVisible(true);
        gui.setTitle("Hype train");

        File config = new File("config.xml");
        if (!config.exists()) {
            config_handler.create_config();
            setting_data = config_handler.load_config();

            //hard coded!!!
            vol = Integer.parseInt(setting_data[0][1]);
            hyp = Float.parseFloat(setting_data[1][1]);
            sym = setting_data[2][1];
            isSorted = Boolean.parseBoolean(setting_data[3][1]);

            refresh(true, true, true, false);

            Settings_handler gui_Setting = new Settings_handler(vol, hyp, sym = create_sym_array(), isSorted);
            gui_Setting.setVisible(true);
            gui_Setting.setSize(500, 500);
            gui_Setting.setAlwaysOnTop(true);
            System.out.println("New config created");

        } else {
            System.out.println("Load config");
            setting_data = config_handler.load_config();
            vol = Integer.parseInt(setting_data[0][1]);
            hyp = Float.parseFloat(setting_data[1][1]);
            sym = setting_data[2][1];
            isSorted = Boolean.parseBoolean(setting_data[3][1]);

            load_table(sym);

            refresh(true, true, true, false);
            System.out.println("config loaded");
        }
    }

    public static void refresh(boolean symbols, boolean charts, boolean notification, boolean settings) { //Method for refreshing the ui based on the given panels to refresh
        if (symbols) {
            symbol_panel.revalidate();
            symbol_panel.repaint();
        }

        if (charts) {
            chart_tool_panel.revalidate();
            chart_tool_panel.repaint();
        }

        if (notification) {
            hype_panel.revalidate();
            hype_panel.repaint();
        }

        if (settings) {
            Settings_handler.settingsPanel.revalidate();
            Settings_handler.settingsPanel.repaint();
        }

        System.out.println("Refreshed");
    }

    public static void load_config() {
        setting_data = config_handler.load_config();

        //hard coded!!!
        vol = Integer.parseInt(setting_data[0][1]);
        hyp = Float.parseFloat(setting_data[1][1]);
        sym = setting_data[2][1];
        isSorted = Boolean.parseBoolean(setting_data[3][1]);

        refresh(true, true, true, true);

        System.out.println("Config reloaded!");
    }

    public static void save_config(String[][] data) {
        config_handler.save_config(data);
        System.out.println("Config saved successfully");
    }

    public static void load_table(String config) {
        // Split the string into individual entries
        config = config.substring(1, config.length() - 1); // Remove outer brackets
        String[] entries = config.split("],\\[");

        // Create a 2D array to hold the stock symbol and corresponding Color object
        Object[][] stockArray = new Object[entries.length][2]; // 2D array: [stockSymbol, Color]

        // Iterate through each entry and populate the 2D array
        for (int i = 0; i < entries.length; i++) {
            // Split by "," to separate the stock symbol and color part
            String[] parts = entries[i].split(",java.awt.Color\\[r=");
            String stockSymbol = parts[0]; // Get the stock symbol
            String colorString = parts[1]; // Get the color part (e.g., "102,g=205,b=170]")

            // Parse the RGB values
            String[] rgbParts = colorString.replace("]", "").split(",g=|,b=");
            int r = Integer.parseInt(rgbParts[0]);
            int g = Integer.parseInt(rgbParts[1]);
            int b = Integer.parseInt(rgbParts[2]);

            // Create a Color object from the RGB values
            Color color = new Color(r, g, b);

            // Add the stock symbol and color to the 2D array
            stockArray[i][0] = stockSymbol;
            stockArray[i][1] = color;
        }

        for (Object[] objects : stockArray) {
            stockListModel.addElement(objects[0].toString());
            stockColors.put(objects[0].toString(), (Color) objects[1]);
        }
    }

    public static String create_sym_array() {
        StringBuilder symBuilder = new StringBuilder();

        for (Map.Entry<String, Color> entry : stockColors.entrySet()) {
            String stockSymbol = entry.getKey(); // Get the key (stock symbol)
            Color color = entry.getValue();      // Get the value (color)

            symBuilder.append("[").append(stockSymbol).append(",").append(color).append("],");
        }

        // Remove the trailing comma if the StringBuilder is not empty
        if (symBuilder.length() > 0) {
            symBuilder.setLength(symBuilder.length() - 1); // Remove the last comma
        }
        return symBuilder.toString();
    }

    public JPanel create_symbol_panel() {
        // Create a panel with BorderLayout
        JPanel panel = new JPanel(new BorderLayout());
        panel.setPreferredSize(new Dimension(250, 0)); // Set fixed width of 150px

        // Create a search field at the top
        searchField = new JTextField();
        searchField.setBorder(BorderFactory.createTitledBorder("Search"));

        // Create a list for possible search results
        DefaultListModel<String> searchListModel = new DefaultListModel<>();
        JList<String> searchList = new JList<>(searchListModel);
        searchList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        JScrollPane searchScrollPane = new JScrollPane(searchList);

        searchScrollPane.setPreferredSize(new Dimension(125, 0));

        // Create a scrollable list of stock items using DefaultListModel
        stockListModel = new DefaultListModel<>();
        JList<String> stockList = new JList<>(stockListModel);
        stockList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // Define fixed colors for each stock symbol using a HashMap
        stockColors = new HashMap<>();

        // Create a custom ListCellRenderer to apply fixed colors and round borders
        stockList.setCellRenderer((list, value, index, isSelected, cellHasFocus) -> {
            JLabel label = new JLabel(value, JLabel.CENTER); // Center text

            // Set fixed background color from the map
            Color fixedColor = stockColors.getOrDefault(value, Color.LIGHT_GRAY); // Default to light gray if no color is mapped
            label.setOpaque(true);
            label.setBackground(fixedColor);

            // Set black round border
            label.setBorder(BorderFactory.createLineBorder(Color.BLACK, 2, true));

            // Handle selection styling without changing the background color
            if (isSelected) {
                label.setBackground(fixedColor); // Keep the fixed color
                label.setForeground(Color.WHITE); // Change text color on selection
                selected_stock = value; //assign the symbol to the variable to extract it

                refreshChartData(1); //add the initial company chart to the screen
            } else {
                label.setForeground(Color.BLACK); // Default text color when not selected
            }

            return label;
        });

        // Add the stock list into a scroll pane
        JScrollPane stockScrollPane = new JScrollPane(stockList);

        // Create the "-" button for removing selected items
        removeButton = new JButton("-");
        removeButton.addActionListener(e -> {
            // Get the selected value
            String selectedValue = stockList.getSelectedValue();
            if (selectedValue != null) {
                // Remove the selected value from the list model
                stockColors.remove(selectedValue);
                stockListModel.removeElement(selectedValue);

                sym = create_sym_array(); //Create the symbol array
            }
        });

        // Create a sub-panel for the button at the bottom
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        buttonPanel.add(removeButton);

        addButton = new JButton("+");
        buttonPanel.add(addButton);
        addButton.addActionListener(e -> {
            String selectedSymbol = searchList.getSelectedValue();
            if (selectedSymbol != null && !stockListModel.contains(selectedSymbol)) {
                // Add the selected symbol to the stock list
                stockListModel.addElement(selectedSymbol);
                stockColors.put(selectedSymbol, generateRandomColor()); // Assign a random color or use another logic
                sym = create_sym_array();
            }
        });

        // Add the search field to the top, the scrollable stock list to the center, and the button panel to the bottom
        panel.add(searchField, BorderLayout.NORTH);
        panel.add(stockScrollPane, BorderLayout.CENTER);
        panel.add(buttonPanel, BorderLayout.SOUTH);

        // Add the search list for suggestions (optional)
        panel.add(searchScrollPane, BorderLayout.EAST);

        // Optional: Add a border with title to the panel
        panel.setBorder(BorderFactory.createTitledBorder("Stock Symbols"));

        // Add a document listener to update the search list dynamically
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

            private void updateSearchList() {
                String searchText = searchField.getText().trim();
                searchListModel.clear();

                if (!searchText.isEmpty()) {
                    // Filter or search logic to populate searchListModel with matching symbols
                    List<String> matchedSymbols = Main_data_handler.findMatchingSymbols(searchText); // Implement this method
                    for (String symbol : matchedSymbols) {
                        searchListModel.addElement(symbol);
                    }
                }
            }
        });

        return panel;
    }

    private Color generateRandomColor() {
        Random rand = new Random();
        return new Color(rand.nextInt(256), rand.nextInt(256), rand.nextInt(256));
    }

    public JPanel create_chart_tool_panel() {
        JPanel mainPanel = new JPanel(new BorderLayout());

        // First Row - Chart and News
        JPanel firstRowPanel = new JPanel(new BorderLayout());

        // Left side of the first row - Chart Panel with Buttons above
        chartPanel = new JPanel(new BorderLayout());

        // Buttons for time range selection (Day, Week, Month)
        JPanel buttonPanel = new JPanel();
        oneDayButton = new JButton("1 Day");
        threeDaysButton = new JButton("3 Days");
        oneWeekButton = new JButton("1 Week");
        twoWeeksButton = new JButton("2 Weeks");
        oneMonthButton = new JButton("1 Month");
        threeMonthsButton = new JButton("3 Months");
        sixMonthsButton = new JButton("6 Months");
        oneYearButton = new JButton("1 Year");

        buttonPanel.add(oneDayButton);
        buttonPanel.add(threeDaysButton);
        buttonPanel.add(oneWeekButton);
        buttonPanel.add(twoWeeksButton);
        buttonPanel.add(oneMonthButton);
        buttonPanel.add(threeMonthsButton);
        buttonPanel.add(sixMonthsButton);
        buttonPanel.add(oneYearButton);

        // Adding Action Listeners with lambda expressions
        oneDayButton.addActionListener(e -> refreshChartData(1));
        threeDaysButton.addActionListener(e -> refreshChartData(2));
        oneWeekButton.addActionListener(e -> refreshChartData(3));
        twoWeeksButton.addActionListener(e -> refreshChartData(4));
        oneMonthButton.addActionListener(e -> refreshChartData(5));
        threeMonthsButton.addActionListener(e -> refreshChartData(6));
        sixMonthsButton.addActionListener(e -> refreshChartData(7));
        oneYearButton.addActionListener(e -> refreshChartData(8));

        // Placeholder for JFreeChart (replace with actual chart code)
        JPanel chartPlaceholder = new JPanel(new BorderLayout());
        chartPlaceholder.setBorder(BorderFactory.createLineBorder(Color.BLACK));
        chartPlaceholder.setPreferredSize(new Dimension(600, 400));

        // Initialize the chart
        timeSeries = new TimeSeries(selected_stock + " price");
        chartDisplay = createChart(timeSeries, selected_stock + " price Chart");

        // Add a titled border to the chart panel
        chartPanel.setBorder(BorderFactory.createTitledBorder("Stock price Chart"));
        chartPanel.add(buttonPanel, BorderLayout.NORTH);
        chartPanel.add(chartPlaceholder, BorderLayout.CENTER);
        chartPanel.add(chartDisplay);

        // Right side of the first row - Company News
        // Initialize the News list model
        NewsListModel = new DefaultListModel<>();
        NewsList = new JList<>(NewsListModel);
        NewsList.setVisibleRowCount(10); // Set the visible row count
        NewsList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // Add mouse listener to handle clicks
        NewsList.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (e.getClickCount() == 2) { // Open on double click
                    int index = NewsList.locationToIndex(e.getPoint());
                    if (index != -1) {
                        News clickedNews = NewsListModel.getElementAt(index);
                        openNews(clickedNews);
                    }
                }
            }
        });

        JScrollPane newsScrollPane = new JScrollPane(NewsList);
        newsScrollPane.setPreferredSize(new Dimension(200, 400));

        // Add a titled border to the news section
        newsScrollPane.setBorder(BorderFactory.createTitledBorder("Company News"));

        // Add chartPanel (80%) and newsScrollPane (20%) to the firstRowPanel
        firstRowPanel.add(chartPanel, BorderLayout.CENTER);
        firstRowPanel.add(newsScrollPane, BorderLayout.EAST);

        // Second Row - Lists of stock information
        JPanel secondRowPanel = new JPanel(new GridLayout(1, 4));

        // First column - Open, High, Low
        JPanel openHighLowPanel = new JPanel(new GridLayout(3, 1));
        openLabel = new JLabel("Open: ");
        highLabel = new JLabel("High: ");
        lowLabel = new JLabel("Low: ");
        openHighLowPanel.add(openLabel);
        openHighLowPanel.add(highLabel);
        openHighLowPanel.add(lowLabel);

        // Second column - Volume, P/E, Market Cap
        JPanel volumePEMktCapPanel = new JPanel(new GridLayout(3, 1));
        volumeLabel = new JLabel("Vol: ");
        peLabel = new JLabel("P/E: ");
        mktCapLabel = new JLabel("Mkt Cap: ");
        volumePEMktCapPanel.add(volumeLabel);
        volumePEMktCapPanel.add(peLabel);
        volumePEMktCapPanel.add(mktCapLabel);

        // Third column - 52W High, 52W Low, Avg Volume
        JPanel rangeAndAvgVolPanel = new JPanel(new GridLayout(3, 1));
        fiftyTwoWkHighLabel = new JLabel("52W H: ");
        fiftyTwoWkLowLabel = new JLabel("52W L: ");
        avgVolumeLabel = new JLabel("Avg Vol: ");
        rangeAndAvgVolPanel.add(fiftyTwoWkHighLabel);
        rangeAndAvgVolPanel.add(fiftyTwoWkLowLabel);
        rangeAndAvgVolPanel.add(avgVolumeLabel);

        // Add all columns to the secondRowPanel
        secondRowPanel.add(openHighLowPanel);
        secondRowPanel.add(volumePEMktCapPanel);
        secondRowPanel.add(rangeAndAvgVolPanel);

        // Add a titled border to the stock info section
        secondRowPanel.setBorder(BorderFactory.createTitledBorder("Stock Information"));

        // Add first and second rows to the main panel
        mainPanel.add(firstRowPanel, BorderLayout.CENTER);
        mainPanel.add(secondRowPanel, BorderLayout.SOUTH);

        return mainPanel;
    }

    public void updateStockInfoLabels(double open, double high, double low, double volume, double peRatio, double marketCap, double fiftyTwoWkHigh, double fiftyTwoWkLow, double avgVolume) {
        openLabel.setText("Open: " + String.format("%.2f", open));
        highLabel.setText("High: " + String.format("%.2f", high));
        lowLabel.setText("Low: " + String.format("%.2f", low));
        volumeLabel.setText("Vol: " + String.format("%.0f", volume));
        peLabel.setText("P/E: " + String.format("%.2f", peRatio));
        mktCapLabel.setText("Mkt Cap: " + String.format("%.2f", marketCap));
        fiftyTwoWkHighLabel.setText("52W H: " + String.format("%.2f", fiftyTwoWkHigh));
        fiftyTwoWkLowLabel.setText("52W L: " + String.format("%.2f", fiftyTwoWkLow));
        avgVolumeLabel.setText("Avg Vol: " + String.format("%.0f", avgVolume));
    }

    public void refreshChartData(int choice) {
        //Create a new time series
        timeSeries = new TimeSeries(selected_stock + " price");

        // Clear the existing data in the TimeSeries
        timeSeries.clear();


        //!!!Add the real chart logic to it
        for (int i = 0; i < choice * 100; i++) {
            // Add a new point to the time series for the last 10 seconds
            timeSeries.add(new Second(new java.util.Date(System.currentTimeMillis() - i * 1000L)), Math.random() * 10);
        }


        switch (choice) { //switch the cases for the different time periods
            case 1: {

                break;
            }
            case 2: {

                break;
            }
            case 3: {

                break;
            }
            case 4: {

                break;
            }
            case 5: {

                break;
            }
            case 6: {

                break;
            }
            case 7: {

                break;
            }
            case 8: {

                break;
            }
            default: {
                break;
            }
        }

        //updateStockInfoLabels(); //!!!add the stock detail later after extracting

        // Create a new chart with the updated title
        ChartPanel newChartDisplay = createChart(timeSeries, selected_stock + " Price Chart");

        // Remove the old chart
        chartPanel.remove(chartDisplay); // Remove the old chart
        chartDisplay = newChartDisplay; // Update the reference to the new chart
        chartPanel.add(chartDisplay, BorderLayout.CENTER); // Add the new chart

        chartPanel.revalidate(); // Refresh the panel
        chartPanel.repaint(); // Repaint the panel to show new chart
    }

    private ChartPanel createChart(TimeSeries timeSeries, String chartName) {
        // Wrap the TimeSeries in a TimeSeriesCollection
        TimeSeriesCollection dataset = new TimeSeriesCollection();
        dataset.addSeries(timeSeries);

        // Create the chart with the dataset
        JFreeChart chart = ChartFactory.createTimeSeriesChart(
                chartName, // Chart title
                "Date", // X-axis Label
                "Price", // Y-axis Label
                dataset, // The dataset
                true, // Show legend
                true, // Show tooltips
                false // Show URLs
        );

        // Customizing the plot
        XYPlot plot = chart.getXYPlot();
        plot.addRangeMarker(new IntervalMarker(Double.NEGATIVE_INFINITY, 0.0, new Color(255, 200, 200, 100)));
        plot.addRangeMarker(new IntervalMarker(0.0, Double.POSITIVE_INFINITY, new Color(200, 255, 200, 100)));

        // Add a black line at y = 0
        ValueMarker zeroLine = new ValueMarker(0.0);
        zeroLine.setPaint(Color.BLACK);
        zeroLine.setStroke(new BasicStroke(1.0f));
        plot.addRangeMarker(zeroLine);

        return new ChartPanel(chart);
    }

    public JPanel create_hype_panel() {
        JPanel panel = new JPanel();
        panel.setPreferredSize(new Dimension(300, 0)); // Set fixed width of 200px
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        JLabel notifications = new JLabel("Hype notifications");
        panel.add(notifications);

        // Initialize the notification list model
        notificationListModel = new DefaultListModel<>();
        notificationList = new JList<>(notificationListModel);
        notificationList.setVisibleRowCount(10); // Set the visible row count
        notificationList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // Add mouse listener to handle clicks
        notificationList.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (isSorted) {
                    sort_notifications();
                }

                if (e.getClickCount() == 2) { // Open on double click
                    int index = notificationList.locationToIndex(e.getPoint());
                    if (index != -1) {
                        Notification clickedNotification = notificationListModel.getElementAt(index);
                        openNotification(clickedNotification);
                    }
                }
            }
        });

        // Wrap the JList in a JScrollPane
        JScrollPane scrollPane = new JScrollPane(notificationList);
        scrollPane.setPreferredSize(new Dimension(200, 100)); // Set the preferred size of the scroll pane
        panel.add(scrollPane);

        // Optional: Add some spacing between components
        panel.add(Box.createRigidArea(new Dimension(0, 10))); // Vertical spacing

        panel.setBorder(BorderFactory.createTitledBorder("Notifications"));

        //test button to add test notifications
        JButton button = new JButton("add (test)");
        panel.add(button);
        button.addActionListener(new event_addNotification());

        return panel;
    }

    // Method to add a notification
    public void addNotification(String title, String content, TimeSeries timeSeries) { //Method to add a new notification to the panel
        Notification newNotification = new Notification(title, content, timeSeries);
        notificationListModel.addElement(newNotification);
    }

    // Open notification and close previous one
    private void openNotification(Notification notification) {
        // Close the currently opened notification if it exists
        if (currentNotification != null) {
            currentNotification.closeNotification();
        }
        // Show the new notification
        notification.showNotification();
        currentNotification = notification; // Update the current notification reference
    }

    // Method to add a News
    public void addNews(String title, String content) { //Method to add News to the panel
        News newNews = new News(title, content);
        NewsListModel.addElement(newNews);
    }

    // Open News and close previous one
    private void openNews(News news) {
        // Close the currently opened News if it exists
        if (CurrentNews != null) {
            CurrentNews.closeNews();
        }

        news.showNews();
        CurrentNews = news;
    }

    // Create the menu bar
    private JMenuBar createMenuBar() {
        menuBar = new JMenuBar();

        // Create menus
        file = new JMenu("File");
        settings = new JMenu("Settings");
        hype_mode_menu = new JMenu("Hype mode");
        Notifications = new JMenu("Notifications");

        //JMenuItems File
        load = new JMenuItem("Load the config (manually again)");
        save = new JMenuItem("Save the config");
        exit = new JMenuItem("Exit and don't save");

        //Settings
        setting_handler = new JMenuItem("Open settings");

        //Hype mode
        activate_hype_mode = new JMenuItem("Activate hype mode");

        //Notifications
        clear = new JMenuItem("Clear Notifications");
        sort = new JMenuItem("Sort Notifications");

        //add it to the menus File
        file.add(load);
        file.add(save);
        file.add(exit);

        //Settings
        settings.add(setting_handler);

        //HypeMode
        hype_mode_menu.add(activate_hype_mode);

        //Notifications
        Notifications.add(clear);
        Notifications.add(sort);

        // Add menus to the menu bar
        menuBar.add(file);
        menuBar.add(settings);
        menuBar.add(hype_mode_menu);
        menuBar.add(Notifications);

        load.addActionListener(new event_Load());
        save.addActionListener(new event_save());
        exit.addActionListener(new event_exit());
        setting_handler.addActionListener(new event_settings());
        activate_hype_mode.addActionListener(new event_activate_hype_mode());
        clear.addActionListener(e -> notificationListModel.clear());
        sort.addActionListener(new event_sort_notifications());

        return menuBar;
    }

    // Helper method to extract percentage value from the notification title
    public float extractPercentage(String title) {
        int percentIndex = title.indexOf("%");
        if (percentIndex != -1) {
            try {
                // Parse the substring into a float value
                return Float.parseFloat(title.substring(0, percentIndex));
            } catch (NumberFormatException e) {
                e.printStackTrace(); // Handle invalid format gracefully
            }
        }
        return 0.0f; // Default to 0 if extraction fails
    }

    public void sort_notifications() {
        // Convert the notification list model to a List for easier manipulation
        List<Notification> notifications = new ArrayList<>();

        for (int i = 0; i < notificationListModel.size(); i++) {
            notifications.add(notificationListModel.getElementAt(i));
        }

        // Sort the notifications based on percentage change
        notifications.sort((n1, n2) -> {
            // Extract the percentage from each notification title
            float percent1 = extractPercentage(n1.getTitle());
            float percent2 = extractPercentage(n2.getTitle());

            // Sort in descending order of percentage
            return Float.compare(percent2, percent1);
        });

        // Clear the model and repopulate it with sorted notifications
        notificationListModel.clear();
        for (Notification notification : notifications) {
            notificationListModel.addElement(notification);
        }
    }

    public static class event_activate_hype_mode implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            System.out.println("Activating hype mode for auto stock scanning");
            Main_data_handler.start_Hype_Mode(vol, hyp);
        }
    }

    public static class event_settings implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            Settings_handler gui = new Settings_handler(vol, hyp, sym, isSorted);
            gui.setSize(500, 500);
            gui.setAlwaysOnTop(true);
            gui.setTitle("Config handler ");
            gui.setVisible(true);
        }
    }

    public static class event_Load implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            load_config();
        }
    }

    public static class event_save implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            System.out.println(sym);
            String[][] values = {
                    {"volume", String.valueOf(vol)},
                    {"hype_strength", String.valueOf(hyp)},
                    {"symbols", sym = create_sym_array()},
                    {"sort", String.valueOf(isSorted)}
            };

            save_config(values);
        }
    }

    public static class event_exit implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            System.out.println("Exit application");
            System.exit(0); // Exit the application
        }
    }

    public class event_sort_notifications implements ActionListener {
        @Override
        public void actionPerformed(ActionEvent e) {
            sort_notifications();
        }
    }

    //test method
    public class event_addNotification implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                List<Notification> notifications = data_tester.Main_data_puller();

                for (Notification notification : notifications) {
                    addNotification(notification.getTitle(), notification.getContent(), notification.getTimeSeries()); //add notification sample
                }

                addNews(String.valueOf(Math.random() * 10), String.valueOf(Math.random() * 10)); //add news sample

            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
        }
    }
}
//TODO
//!!!Add the real chart logic to it to the chart panel
//!!!add the stock detail later after extracting