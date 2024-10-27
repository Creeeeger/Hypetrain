package org.crecker;

import javax.swing.*;
import java.awt.*;

public class News {
    private final String title;
    private final String content;
    private JFrame NewsFrame; // Frame for the News

    public News(String title, String content) {
        this.title = title;
        this.content = content;
    }

    public void showNews() {
        // Create the News window
        NewsFrame = new JFrame(title);
        NewsFrame.setSize(600, 400); // Adjust the size for both text and chart
        NewsFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        NewsFrame.setLocationRelativeTo(null);
        NewsFrame.setAlwaysOnTop(true);

        // Create the text area with content, enable line wrapping
        JTextArea textArea = new JTextArea(content);
        textArea.setEditable(false);
        textArea.setLineWrap(true);         // Enable line wrapping
        textArea.setWrapStyleWord(true);    // Wrap at word boundaries

        // Add the text area to a scroll pane
        JScrollPane scrollPane = new JScrollPane(textArea);

        // Create the panel to hold text
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout());

        // Add the scroll pane with the text area to the top of the panel
        mainPanel.add(scrollPane, BorderLayout.NORTH);
        NewsFrame.add(mainPanel);
        NewsFrame.setVisible(true);
    }

    public void closeNews() {
        if (NewsFrame != null) {
            NewsFrame.dispose();
        }
    }

    @Override
    public String toString() {
        return title; // Display the title in the list
    }
}