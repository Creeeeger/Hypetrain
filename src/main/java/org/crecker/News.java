package org.crecker;

import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.net.URI;

/**
 * <h1>News</h1>
 * Represents a news item with a title, content, and a clickable URL.
 * Handles the display of the news in a pop-up JFrame, including clickable hyperlink functionality.
 * Designed for desktop Java applications using Swing.
 */
public class News {

    // --- Fields ---
    private final String title;   // The headline/title of the news article
    private final String content; // The detailed content or summary of the news article
    private final String url;     // A URL linking to the full news story
    private JFrame NewsFrame;     // The Swing window used to display this news item

    /**
     * Constructs a News object.
     *
     * @param title   the news title/headline
     * @param content the news story or summary
     * @param url     the URL for the full article (as a clickable link)
     */
    public News(String title, String content, String url) {
        this.title = title;
        this.content = content;
        this.url = url;
    }

    /**
     * Displays this News item in a new JFrame with its title, content, and clickable link.
     * The window will be centered, always on top, and properly sized for readability.
     */
    public void showNews() {
        // Initialize the JFrame with the news title as the window title
        NewsFrame = new JFrame(title);
        NewsFrame.setSize(600, 400); // Suitable size for both the text area and controls
        NewsFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); // Only close this window
        NewsFrame.setLocationRelativeTo(null);  // Center the window on the screen
        NewsFrame.setAlwaysOnTop(true);         // Keep this window above others for visibility

        // Generate a clickable URL label at the top of the window
        JLabel urlLabel = getURL();

        // Set up the text area for the main news content (word-wrapped, not editable)
        JTextArea textArea = new JTextArea(content);
        textArea.setEditable(false);       // Prevent editing by the user
        textArea.setLineWrap(true);        // Enable wrapping to avoid horizontal scroll
        textArea.setWrapStyleWord(true);   // Wrap at word boundaries for readability

        // Place the text area inside a scroll pane in case the content is long
        JScrollPane scrollPane = new JScrollPane(textArea);

        // Main panel uses BorderLayout: URL at the top, content in the center
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BorderLayout());
        mainPanel.add(urlLabel, BorderLayout.NORTH);     // Clickable link at top
        mainPanel.add(scrollPane, BorderLayout.CENTER);  // Scrollable news content

        // Add the panel to the frame and display the window
        NewsFrame.add(mainPanel);
        NewsFrame.setVisible(true);
    }

    /**
     * Creates a JLabel displaying the news URL as a clickable hyperlink.
     * Opens the link in the user's default browser on click.
     *
     * @return JLabel configured as a clickable URL
     */
    @NotNull
    private JLabel getURL() {
        // JLabel with HTML to make it appear like a link
        JLabel urlLabel = new JLabel("<html><a href=''>" + url + "</a></html>");
        urlLabel.setCursor(new Cursor(Cursor.HAND_CURSOR)); // Show hand cursor to indicate it's clickable

        // Attach a mouse listener to handle the click event
        urlLabel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                try {
                    // Try to open the URL in the default desktop browser
                    Desktop.getDesktop().browse(new URI(url));
                } catch (Exception ex) {
                    ex.printStackTrace(); // If unable to open (bad URL or permissions), print the stack trace
                }
            }
        });
        return urlLabel;
    }

    /**
     * Closes the News JFrame if it is currently open.
     * Can be called to programmatically dismiss the news window.
     */
    public void closeNews() {
        if (NewsFrame != null) {
            NewsFrame.dispose(); // Cleanly dispose of the frame and free resources
        }
    }

    /**
     * Returns the news title as the string representation.
     * This is useful for displaying the News object in UI lists.
     *
     * @return the news title
     */
    @Override
    public String toString() {
        return title;
    }

    // --- Getters ---

    /**
     * @return the news title
     */
    public String getTitle() {
        return title;
    }

    /**
     * @return the news content
     */
    public String getContent() {
        return content;
    }
}