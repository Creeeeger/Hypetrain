package org.crecker;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class Main_UI extends JFrame {
    static int vol;
    static float hyp;
    static String[][] setting_data;
    static JPanel symbol_panel, chart_tool_panel, hype_panel;
    static JMenuBar menuBar;
    static JMenu file, settings;
    static JMenuItem load, save, exit, setting_handler;

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

            System.out.println(vol + " " + hyp);
            Settings_handler gui_Setting = new Settings_handler(vol, hyp);
            gui_Setting.setVisible(true);
            gui_Setting.setSize(500, 500);
            gui_Setting.setAlwaysOnTop(true);
            System.out.println("New config created");

        } else {
            System.out.println("Load config");
            setting_data = config_handler.load_config();

            for (int i = 0; i < setting_data.length; i++) {
                System.out.println(setting_data[i][1]);
            }

            System.out.println("config loaded");
        }
    }

    private JPanel create_symbol_panel() {
        // Create a panel with BorderLayout
        JPanel panel = new JPanel(new BorderLayout());
        panel.setPreferredSize(new Dimension(150, 0)); // Set fixed width of 150px

        // Create a search field at the top
        JTextField searchField = new JTextField();
        searchField.setBorder(BorderFactory.createTitledBorder("Search"));

        // Create a scrollable list of stock items
        String[] stockItems = {"NVDA", "AAPL", "GOOGL", "TSLA", "MSFT"};
        JList<String> stockList = new JList<>(stockItems);
        stockList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        // Define fixed colors for each stock symbol using a HashMap
        Map<String, Color> stockColors = new HashMap<>();
        stockColors.put("NVDA", new Color(102, 205, 170)); // Medium Aquamarine
        stockColors.put("AAPL", new Color(135, 206, 250)); // Light Sky Blue
        stockColors.put("GOOGL", new Color(255, 182, 193)); // Light Pink
        stockColors.put("TSLA", new Color(240, 230, 140)); // Khaki
        stockColors.put("MSFT", new Color(221, 160, 221)); // Plum

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

        // Add the search field to the top and the scrollable stock list to the center of the panel
        panel.add(searchField, BorderLayout.NORTH);
        panel.add(scrollPane, BorderLayout.CENTER);

        // Optional: Add a border with title to the panel
        panel.setBorder(BorderFactory.createTitledBorder("Stock Symbols"));

        return panel;
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

    private JPanel create_hype_panel() {
        JPanel panel = new JPanel();
        panel.setPreferredSize(new Dimension(200, 0)); // Set fixed width of 150px
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        JLabel notifications = new JLabel("Hype notifications");
        panel.add(notifications);

        // Optional: Add some spacing between tools
        panel.add(Box.createRigidArea(new Dimension(0, 10))); // Vertical spacing

        panel.setBorder(BorderFactory.createTitledBorder("Notifications"));

        return panel;
    }

    // Create the menu bar
    private JMenuBar createMenuBar() {
        menuBar = new JMenuBar();

        // Create menus
        file = new JMenu("File");
        settings = new JMenu("Settings");

        //JMenuItems
        load = new JMenuItem("Load the config");
        save = new JMenuItem("Save the config");
        exit = new JMenuItem("Exit and don't save");

        setting_handler = new JMenuItem("Open settings");

        //add it to the menus
        file.add(load);
        file.add(save);
        file.add(exit);

        settings.add(setting_handler);

        // Add menus to the menu bar
        menuBar.add(file);
        menuBar.add(settings);

        load.addActionListener(new event_Load());
        save.addActionListener(new event_save());
        exit.addActionListener(new event_exit());
        setting_handler.addActionListener(new event_settings());

        return menuBar;
    }

    public class event_Load implements ActionListener{

        @Override
        public void actionPerformed(ActionEvent e) {

        }
    }

    public class event_save implements ActionListener{

        @Override
        public void actionPerformed(ActionEvent e) {

        }
    }

    public class event_exit implements ActionListener{

        @Override
        public void actionPerformed(ActionEvent e) {

        }
    }

    public class event_settings implements ActionListener{

        @Override
        public void actionPerformed(ActionEvent e) {

        }
    }
}
