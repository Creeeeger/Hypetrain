package org.crecker;

import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.net.URI;

public class News {

    private final String title;
    private final String content;
    private final String url;
    private JFrame NewsFrame; // Frame for the News

    public News(String title, String content, String url) {
        this.title = title;
        this.content = content;
        this.url = url;
    }

    public void showNews() {
        // Create the News window
        NewsFrame = new JFrame(title);
        NewsFrame.setSize(600, 400); // Adjust the size for both text and chart
        NewsFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        NewsFrame.setLocationRelativeTo(null);
        NewsFrame.setAlwaysOnTop(true);

        // Create the clickable URL as a JLabel with HTML
        JLabel urlLabel = getURL();

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

        // Add the clickable link (URL) to the top
        mainPanel.add(urlLabel, BorderLayout.NORTH);

        // Add the scroll pane with the text area below the URL
        mainPanel.add(scrollPane, BorderLayout.CENTER);

        // Add the main panel to the frame
        NewsFrame.add(mainPanel);
        NewsFrame.setVisible(true);
    }

    @NotNull
    private JLabel getURL() {
        JLabel urlLabel = new JLabel("<html><a href=''>" + url + "</a></html>");
        urlLabel.setCursor(new Cursor(Cursor.HAND_CURSOR));

        // Add a click listener to open the link in a browser
        urlLabel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                try {
                    Desktop.getDesktop().browse(new URI(url));
                } catch (Exception ex) {
                    System.out.println(ex.getMessage());
                }
            }
        });
        return urlLabel;
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