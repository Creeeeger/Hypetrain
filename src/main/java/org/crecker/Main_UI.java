package org.crecker;

import org.jfree.data.time.TimeSeries;

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
    static String sym;
    static boolean isSorted;
    static String[][] setting_data;
    static JPanel symbol_panel, chart_tool_panel, hype_panel;
    static JMenuBar menuBar;
    static JTextField searchField;
    static JButton removeButton, addButton;
    static JMenu file, settings, hype_mode_menu, Notifications;
    static JMenuItem load, save, exit, setting_handler, activate_hype_mode, clear, sort;
    static DefaultListModel<String> stockListModel;
    static Map<String, Color> stockColors;
    private DefaultListModel<Notification> notificationListModel;
    private JList<Notification> notificationList;
    private Notification currentNotification; // Track currently opened notification

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
        gui.setSize(1000, 800); // Width and height of the window
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

    private JPanel create_chart_tool_panel() {
        JPanel panel = new JPanel(new BorderLayout());

        // Display stock courses in a text area
        JTextArea stockCourses = new JTextArea("Stock Course Information...\nSelect a stock to view details.");
        stockCourses.setEditable(false); // Make the text area read-only
        JScrollPane scrollPane = new JScrollPane(stockCourses);

        panel.add(scrollPane, BorderLayout.CENTER);
        panel.setBorder(BorderFactory.createTitledBorder("Stock Chats"));

        return panel;
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
            }
        });

        // Add mouse listener to handle clicks
        notificationList.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
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
    public void addNotification(String title, String content, TimeSeries timeSeries) {
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
                    addNotification(notification.getTitle(), notification.getContent(), notification.getTimeSeries());
                }

            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
        }
    }
}
