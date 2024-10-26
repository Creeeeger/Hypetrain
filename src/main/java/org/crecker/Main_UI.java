package org.crecker;

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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Main_UI extends JFrame {
    static int vol;
    static float hyp;
    static String sym, sym_to_add;
    static String[][] setting_data;
    static JPanel symbol_panel, chart_tool_panel, hype_panel;
    static JMenuBar menuBar;
    static JTextField searchField;
    static JButton removeButton, addButton;
    static JMenu file, settings, hype_mode_menu;
    static JMenuItem load, save, exit, setting_handler, activate_hype_mode;
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

            refresh(true,true,true,false);

            Settings_handler gui_Setting = new Settings_handler(vol, hyp, sym = create_sym_array());
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
            load_table(sym);

            System.out.println(vol + " " + hyp + " " + sym); //Debug values
            refresh(true,true,true,false);
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
        refresh(true,true,true,true);

        System.out.println(vol + " " + hyp + " " + sym); //Debug values
        System.out.println("Config reloaded!");
    }

    public static void save_config(String[][] data) {
        config_handler.save_config(data);
        System.out.println("Config saved successfully");

    }

    public JPanel create_symbol_panel() {
        // Create a panel with BorderLayout
        JPanel panel = new JPanel(new BorderLayout());
        panel.setPreferredSize(new Dimension(150, 0)); // Set fixed width of 150px

        // Create a search field at the top
        searchField = new JTextField();
        searchField.setBorder(BorderFactory.createTitledBorder("Search"));

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
        JScrollPane scrollPane = new JScrollPane(stockList);

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
        addButton.addActionListener( e -> {
            //Get the search box
            
        });

        // Add the search field to the top, the scrollable stock list to the center, and the button panel to the bottom
        panel.add(searchField, BorderLayout.NORTH);
        panel.add(scrollPane, BorderLayout.CENTER);
        panel.add(buttonPanel, BorderLayout.SOUTH);

        // Optional: Add a border with title to the panel
        panel.setBorder(BorderFactory.createTitledBorder("Stock Symbols"));

        searchField.getDocument().addDocumentListener(new event_change_search());

        return panel;
    }

    public static void load_table(String config){
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

        for (int i = 0; i < stockArray.length; i++) {
            stockListModel.addElement(stockArray[i][0].toString());
            stockColors.put(stockArray[i][0].toString(), (Color) stockArray[i][1]);
        }
    }

    public static String create_sym_array(){
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
        System.out.println(symBuilder);
        return symBuilder.toString();
    }

    public static class event_change_search implements DocumentListener{
        @Override
        public void insertUpdate(DocumentEvent e) {
            sym_to_add = Main_data_handler.sym_to_search(searchField.getText());


            sym = create_sym_array();
        }

        @Override
        public void removeUpdate(DocumentEvent e) {
            sym_to_add = Main_data_handler.sym_to_search(searchField.getText());


            sym = create_sym_array();
        }

        @Override
        public void changedUpdate(DocumentEvent e) {
            sym_to_add = Main_data_handler.sym_to_search(searchField.getText());

            sym = create_sym_array();
        }
    }

    public static void add_Symbol(String symbol){

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
        panel.setPreferredSize(new Dimension(200, 0)); // Set fixed width of 200px
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
    public void addNotification(String title, String content) {
        Notification newNotification = new Notification(title, content);
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

        //JMenuItems
        load = new JMenuItem("Load the config (manually again)");
        save = new JMenuItem("Save the config");
        exit = new JMenuItem("Exit and don't save");

        setting_handler = new JMenuItem("Open settings");

        activate_hype_mode = new JMenuItem("Activate hype mode");

        //add it to the menus
        file.add(load);
        file.add(save);
        file.add(exit);

        settings.add(setting_handler);

        hype_mode_menu.add(activate_hype_mode);

        // Add menus to the menu bar
        menuBar.add(file);
        menuBar.add(settings);
        menuBar.add(hype_mode_menu);

        load.addActionListener(new event_Load());
        save.addActionListener(new event_save());
        exit.addActionListener(new event_exit());
        setting_handler.addActionListener(new event_settings());
        activate_hype_mode.addActionListener(new event_activate_hype_mode());

        return menuBar;
    }

    public static class event_activate_hype_mode implements ActionListener{
        @Override
        public void actionPerformed(ActionEvent e) {
            System.out.println("Activating hype mode for auto stock scanning");

        }
    }

    public static class event_settings implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            Settings_handler gui = new Settings_handler(vol, hyp, sym);
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
                    {"symbols", sym = create_sym_array()}
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

    //test method
    public class event_addNotification implements ActionListener {

        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                List<Notification> notifications = data_tester.Main_data_puller();

                for (Notification notification : notifications) {
                    addNotification(notification.getTitle(), notification.getContent());
                }

            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
        }
    }
}
