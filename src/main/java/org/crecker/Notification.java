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

        // Add content to the window
        JTextArea textArea = new JTextArea(content);
        textArea.setEditable(false);
        notificationFrame.add(new JScrollPane(textArea));

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