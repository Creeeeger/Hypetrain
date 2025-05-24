package org.crecker;

import com.crazzyghost.alphavantage.news.response.NewsResponse;
import org.jetbrains.annotations.NotNull;

import javax.swing.*;
import javax.swing.border.CompoundBorder;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.net.URI;

/**
 * <h1>News</h1>
 * Represents a news item with title, summary, and a clickable URL.
 * Displays news visually, with sentiment color and badge, in a pop-up JFrame for desktop Java Swing apps.
 */
public class News {

    // --- Fields ---
    private final String title;                  // The headline/title of the news article
    private final String content;                // The detailed content or summary of the news article
    private final String url;                    // A URL linking to the full news story
    private JFrame NewsFrame;                    // The Swing window used to display this news item
    private final Color sentimentColor;          // The background color representing news sentiment
    private final NewsResponse.TickerSentiment tickerSentiment; // Sentiment details object (label, score, etc.)

    /**
     * Constructs a News object, with sentiment styling.
     *
     * @param title              the news title/headline
     * @param content            the news story or summary
     * @param url                the URL for the full article (clickable link)
     * @param sentimentForTicker the sentiment data object for this news item (nullable)
     */
    public News(String title, String content, String url, NewsResponse.TickerSentiment sentimentForTicker) {
        this.title = title;
        this.content = content;
        this.url = url;
        this.sentimentColor = sentimentForTicker != null
                ? getColorForSentiment(sentimentForTicker.getSentimentScore()) // Get color by score
                : getColorForSentiment(0.0); // Neutral color if sentiment is unavailable
        this.tickerSentiment = sentimentForTicker;
    }

    /**
     * Displays this News item in a new JFrame, showing the sentiment color bar,
     * title, badge, summary, and a clickable link.
     * The window is centered, always on top, and cannot be resized too small.
     */
    public void showNews() {
        // --- Initialize the JFrame ---
        NewsFrame = new JFrame(title);                       // Set window title
        NewsFrame.setSize(650, 400);                         // Preferred size
        NewsFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); // Only close this news window
        NewsFrame.setLocationRelativeTo(null);               // Center window
        NewsFrame.setAlwaysOnTop(true);                      // Keep on top

        // --- Outer panel: uses BorderLayout for structure ---
        JPanel cardPanel = new JPanel(new BorderLayout());
        cardPanel.setBackground(UIManager.getColor("Panel.background"));
        cardPanel.setBorder(new CompoundBorder(
                new EmptyBorder(18, 20, 18, 20), // Margin inside frame
                new LineBorder(UIManager.getColor("Separator.foreground"), 1, true) // Rounded border
        ));

        // --- Sentiment color bar on the left (WEST) ---
        JPanel colorBar = new JPanel();
        colorBar.setBackground(sentimentColor);              // Set color for sentiment
        colorBar.setPreferredSize(new Dimension(8, 1));      // 8px wide, fills vertically
        cardPanel.add(colorBar, BorderLayout.WEST);

        // --- Main content area (CENTER), vertical stacking ---
        JPanel centerPanel = new JPanel();
        centerPanel.setBackground(cardPanel.getBackground());
        centerPanel.setLayout(new BoxLayout(centerPanel, BoxLayout.Y_AXIS));
        centerPanel.setBorder(BorderFactory.createEmptyBorder(0, 16, 0, 0)); // Space from color bar

        // --- Title (word-wrapped, bold) ---
        JTextArea titleArea = new JTextArea(this.title);
        titleArea.setLineWrap(true);                        // Wrap long titles
        titleArea.setWrapStyleWord(true);
        titleArea.setEditable(false);                       // User can't change text
        titleArea.setOpaque(false);                         // No background color
        titleArea.setFont(titleArea.getFont().deriveFont(Font.BOLD, 17f));
        titleArea.setBorder(BorderFactory.createEmptyBorder(0, 0, 0, 0));
        titleArea.setAlignmentX(Component.LEFT_ALIGNMENT);
        titleArea.setMaximumSize(new Dimension(Integer.MAX_VALUE, 80)); // Allow to expand wide

        // --- Badge for sentiment label (if present) ---
        JLabel badge = null;
        if (tickerSentiment != null) {
            badge = new JLabel(" " + tickerSentiment.getSentimentLabel() + " ");
            badge.setOpaque(true);
            badge.setBackground(sentimentColor);             // Badge uses sentiment color
            badge.setForeground(Color.WHITE);
            badge.setFont(badge.getFont().deriveFont(Font.BOLD, 13f));
            badge.setBorder(BorderFactory.createEmptyBorder(2, 8, 2, 8));
            badge.setAlignmentX(Component.LEFT_ALIGNMENT);
            badge.setMaximumSize(new Dimension(Integer.MAX_VALUE, badge.getPreferredSize().height));
        }

        // --- Summary/Content area (multi-line, word-wrap) ---
        JTextArea summaryArea = new JTextArea(this.content);
        summaryArea.setLineWrap(true);                      // Wrap content
        summaryArea.setWrapStyleWord(true);
        summaryArea.setEditable(false);
        summaryArea.setOpaque(false);
        summaryArea.setFont(summaryArea.getFont().deriveFont(15f));
        summaryArea.setBorder(BorderFactory.createEmptyBorder(12, 0, 12, 0));
        summaryArea.setAlignmentX(Component.LEFT_ALIGNMENT);
        summaryArea.setMaximumSize(new Dimension(Integer.MAX_VALUE, 400)); // Tall

        // --- URL Label (clickable link) ---
        JLabel urlLabel = getURL();
        urlLabel.setAlignmentX(Component.LEFT_ALIGNMENT);

        // --- Add components to central panel, with spacing ---
        centerPanel.add(titleArea);
        if (badge != null) {
            centerPanel.add(Box.createVerticalStrut(4)); // Gap before badge
            centerPanel.add(badge);
        }
        centerPanel.add(Box.createVerticalStrut(12)); // Gap before summary
        centerPanel.add(summaryArea);
        centerPanel.add(Box.createVerticalStrut(12)); // Gap before link
        centerPanel.add(urlLabel);

        // --- Place center panel in cardPanel ---
        cardPanel.add(centerPanel, BorderLayout.CENTER);

        // --- Finalize frame content and display ---
        NewsFrame.setContentPane(cardPanel);
        NewsFrame.setMinimumSize(new Dimension(400, 250)); // Prevent too small resizing
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

    /**
     * Returns a color representation for a sentiment score.
     * <ul>
     *   <li>Bullish sentiment (strong positive): green</li>
     *   <li>Somewhat bullish: light green</li>
     *   <li>Neutral: gray</li>
     *   <li>Somewhat bearish: orange</li>
     *   <li>Bearish (strong negative): red</li>
     * </ul>
     * Color thresholds:
     * <ul>
     *   <li>score ≤ -0.35 : Bearish (red)</li>
     *   <li>score ≤ -0.15 : Somewhat Bearish (orange)</li>
     *   <li>score <  0.15 : Neutral (gray)</li>
     *   <li>score <  0.35 : Somewhat Bullish (light green)</li>
     *   <li>otherwise     : Bullish (green)</li>
     * </ul>
     *
     * @param score the sentiment score (typically between -1 and 1)
     * @return a {@link Color} object reflecting the sentiment polarity and strength
     */
    public static Color getColorForSentiment(double score) {
        if (score <= -0.35) {
            return new Color(200, 50, 50); // Bearish: Red
        } else if (score <= -0.15) {
            return new Color(255, 153, 102); // Somewhat Bearish: Orange
        } else if (score < 0.15) {
            return new Color(200, 200, 200); // Neutral: Grey
        } else if (score < 0.35) {
            return new Color(104, 135, 41); // Somewhat Bullish: Light Green
        } else {
            return new Color(76, 175, 80); // Bullish: Green
        }
    }

    /**
     * Gets the color associated with this news item's sentiment.
     * The color reflects the strength and polarity of sentiment (red for bearish, green for bullish, etc.).
     *
     * @return the sentiment color
     */
    public Color getSentimentColor() {
        return sentimentColor;
    }

    /**
     * Gets the URL pointing to the full news article.
     *
     * @return the article URL as a String
     */
    public String getUrl() {
        return url;
    }
}