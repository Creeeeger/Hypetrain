package org.crecker;

import javax.swing.*;

public class Notification {
    private final String title;
    private final String content;
    private JFrame notificationFrame; // Frame for the notification

    public Notification(String title, String content) {
        this.title = title;
        this.content = content;
    }

    public String getTitle() {
        return title;
    }

    public String getContent() {
        return content;
    }

    public void showNotification() {
        // Create the notification window
        notificationFrame = new JFrame(title);
        notificationFrame.setSize(400, 400);
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
        notificationFrame.add(scrollPane);

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