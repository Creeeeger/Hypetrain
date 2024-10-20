package org.crecker;

import javax.swing.*;
import java.awt.*;

public class Main_UI extends JFrame {

    public static void main(String[] args) {
        Main_UI gui = new Main_UI();
        gui.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        gui.setSize(800, 600); // Width and height of the window
        gui.setVisible(true);
        gui.setTitle("Hype train");
    }

    public Main_UI() {
        // Setting layout for the frame (1 row, 4 columns)
        setLayout(new BorderLayout());
        BorderFactory.createTitledBorder("Stock monitor");

        // Menu bar
        setJMenuBar(createMenuBar());

        // Create the sections
        JPanel leftPanel = createLeftPanel();       // Scroll section with items
        JPanel centerPanel = createCenterPanel();   // Stock courses section
        JPanel toolsPanel = createToolsPanel();     // Tools section
        JPanel notificationsPanel = createNotificationsPanel(); // Notifications section

        // Add sections to the frame using BorderLayout
        add(leftPanel, BorderLayout.WEST);
        add(centerPanel, BorderLayout.CENTER);
        add(toolsPanel, BorderLayout.EAST);
        add(notificationsPanel, BorderLayout.SOUTH); // Using south for better layout management
    }

    // Create the menu bar
    private JMenuBar createMenuBar() {
        JMenuBar menuBar = new JMenuBar();

        // Create menus
        JMenu fileMenu = new JMenu("File");
        JMenu editMenu = new JMenu("Edit");
        JMenu helpMenu = new JMenu("Help");

        // Add menu items to "File" menu
        fileMenu.add(new JMenuItem("Open"));
        fileMenu.add(new JMenuItem("Save"));
        fileMenu.addSeparator(); // Add a separator line
        fileMenu.add(new JMenuItem("Exit"));

        // Add menu items to "Edit" menu
        editMenu.add(new JMenuItem("Undo"));
        editMenu.add(new JMenuItem("Redo"));

        // Add menu items to "Help" menu
        helpMenu.add(new JMenuItem("About"));

        // Add menus to the menu bar
        menuBar.add(fileMenu);
        menuBar.add(editMenu);
        menuBar.add(helpMenu);

        return menuBar;
    }

    // Create the left panel with a scrollable list
    private JPanel createLeftPanel() {
        JPanel panel = new JPanel(new BorderLayout());

        // Create a scrollable list of stock items
        String[] stockItems = {"Apple", "Tesla", "Google", "Amazon", "IBM", "NVIDIA"};
        JList<String> stockList = new JList<>(stockItems);
        stockList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        JScrollPane scrollPane = new JScrollPane(stockList);

        // Add scroll list to the panel
        panel.add(scrollPane, BorderLayout.CENTER);

        // Optional: Add a border with title
        panel.setBorder(BorderFactory.createTitledBorder("Stocks"));

        return panel;
    }

    // Create the center panel for stock courses
    private JPanel createCenterPanel() {
        JPanel panel = new JPanel(new BorderLayout());

        // Display stock courses in a text area
        JTextArea stockCourses = new JTextArea("Stock Course Information...\nSelect a stock to view details.");
        stockCourses.setEditable(false); // Make the text area read-only
        JScrollPane scrollPane = new JScrollPane(stockCourses);

        panel.add(scrollPane, BorderLayout.CENTER);
        panel.setBorder(BorderFactory.createTitledBorder("Stock Courses"));

        return panel;
    }

    // Create the right panel with tools (using a box layout)
    private JPanel createToolsPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));

        // Add tools like buttons and text fields
        panel.add(new JLabel("Tools"));
        panel.add(new JButton("Buy"));
        panel.add(new JButton("Sell"));
        panel.add(new JTextField("Amount"));

        // Optional: Add some spacing between tools
        panel.add(Box.createRigidArea(new Dimension(0, 10))); // Vertical spacing

        panel.setBorder(BorderFactory.createTitledBorder("Tools"));

        return panel;
    }

    // Create the far right panel with notifications
    private JPanel createNotificationsPanel() {
        JPanel panel = new JPanel(new BorderLayout());

        // Notifications displayed in a text area
        JTextArea notificationsArea = new JTextArea("Notifications:\n\n- Stock Alert: AAPL is up 5%\n- New trade executed");
        notificationsArea.setEditable(false); // Make the text area read-only
        JScrollPane scrollPane = new JScrollPane(notificationsArea);

        panel.add(scrollPane, BorderLayout.CENTER);
        panel.setBorder(BorderFactory.createTitledBorder("Notifications"));

        return panel;
    }
}
