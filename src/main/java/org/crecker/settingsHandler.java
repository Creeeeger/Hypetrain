package org.crecker;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import static org.crecker.mainUI.*;

/**
 * settingsHandler is a JFrame-based settings dialog
 * allowing users to configure key application parameters for the stock management system.
 * It provides UI fields for volume, API key, aggressiveness, real-time options, etc.,
 * and writes these settings to the config file on confirmation.
 */
public class settingsHandler extends JDialog {
    // Panel holding all settings UI components
    public static JPanel settingsPanel;
    // Labels and input fields for each setting
    static JLabel volume, infos, sortLabel, keyLabel, realtimeLabel, algoLabel, candleLabel, T212Label, pushCutLabel;
    static JTextField volumeText, keyText, algoAggressivenessText, T212textField, pushCutTextField;
    static JCheckBox sortCheckBox, realtimeBox, candleBox;
    // Internal variables to hold settings values
    int vol;
    float aggressiveness;
    String sym, key, T212, push;
    boolean sort, realtime, useCandles;

    /**
     * Constructs a new settingsHandler dialog with the current config values.
     * Populates the dialog with Swing controls for user input.
     *
     * @param vol            Initial volume value.
     * @param sym            Symbols string (not directly editable here).
     * @param sort           Whether to sort hype entries.
     * @param key            API key.
     * @param realtime       Enable real-time updates.
     * @param aggressiveness Aggressiveness for hype algorithm.
     * @param useCandles     Use candle charts.
     * @param T212           Trading212 Api Key
     * @param push           PushCut URL endpoint
     */
    public settingsHandler(int vol, String sym, boolean sort, String key, boolean realtime, float aggressiveness, boolean useCandles, String T212, String push) {
        // Set layout manager for this JFrame: BorderLayout allows a central content area
        setLayout(new BorderLayout(10, 10));

        // Store all initial config parameters for later use
        this.vol = vol;
        this.sym = sym;
        this.sort = sort;
        this.key = key;
        this.realtime = realtime;
        this.aggressiveness = aggressiveness;
        this.useCandles = useCandles;
        this.T212 = T212;
        this.push = push;

        // Initialize main settings panel with vertical layout (BoxLayout.Y_AXIS stacks items top to bottom)
        settingsPanel = new JPanel();
        settingsPanel.setLayout(new BoxLayout(settingsPanel, BoxLayout.Y_AXIS));
        // Add a titled border to the panel for clarity
        settingsPanel.setBorder(BorderFactory.createTitledBorder("Settings for Stock management"));

        // Informational label at the top of the panel, to guide the user
        infos = new JLabel("Select your settings and then press apply");
        // Add the label to the panel and set its alignment to the left
        infos.setAlignmentX(Component.LEFT_ALIGNMENT);
        settingsPanel.add(infos);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 10))); // Spacer

        // Volume (in USD) label and text field
        volume = new JLabel("Volume in USD:");
        volume.setAlignmentX(Component.LEFT_ALIGNMENT);
        volumeText = new JTextField(String.valueOf(vol), 15);
        volumeText.setAlignmentX(Component.LEFT_ALIGNMENT);

        // Sort label and checkbox for automatic sorting option
        sortLabel = new JLabel("Sort hype entries by change automatically");
        sortLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        sortCheckBox = new JCheckBox();
        sortCheckBox.setSelected(sort);
        sortCheckBox.setAlignmentX(Component.LEFT_ALIGNMENT);

        // API key label and field for entering a premium key if needed
        keyLabel = new JLabel("API key (Premium Key for Hype Mode required):");
        keyLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        keyText = new JTextField(key);
        keyText.setAlignmentX(Component.LEFT_ALIGNMENT);

        // API key label for T212
        T212Label = new JLabel("API key for Trading212:");
        T212Label.setAlignmentX(Component.LEFT_ALIGNMENT);
        T212textField = new JTextField(T212);
        T212textField.setAlignmentX(Component.LEFT_ALIGNMENT);

        // PushCut URL notification endpoint
        pushCutLabel = new JLabel("Url endpoint for your notification on PushCut");
        pushCutLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        pushCutTextField = new JTextField(push);
        pushCutTextField.setAlignmentX(Component.LEFT_ALIGNMENT);

        // Real-time updates label and checkbox
        realtimeLabel = new JLabel("Chart realtime updates (per second)");
        realtimeLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        realtimeBox = new JCheckBox();
        realtimeBox.setSelected(realtime);
        realtimeBox.setAlignmentX(Component.LEFT_ALIGNMENT);

        // Hype algorithm aggressiveness label and field
        algoLabel = new JLabel("<html>Hype aggressiveness<br>(lower number for more but less precise entries, vice versa)</html>");
        algoLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        algoAggressivenessText = new JTextField(String.valueOf(aggressiveness));
        algoAggressivenessText.setAlignmentX(Component.LEFT_ALIGNMENT);

        // Candle chart option label and checkbox
        candleLabel = new JLabel("Use candles instead of a line chart:");
        candleLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        candleBox = new JCheckBox();
        candleBox.setSelected(useCandles);
        candleBox.setAlignmentX(Component.LEFT_ALIGNMENT);

        // Add all components to the settings panel, with spacing for neatness
        settingsPanel.add(volume);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(volumeText);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(sortLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(sortCheckBox);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(keyLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(keyText);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(T212Label);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(T212textField);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(pushCutLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(pushCutTextField);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(realtimeLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(realtimeBox);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(algoLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(algoAggressivenessText);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(candleLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(candleBox);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));

        // "Apply Settings" button for saving and applying configuration
        JButton apply = new JButton("Apply Settings");
        apply.setAlignmentX(Component.LEFT_ALIGNMENT);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Spacer before button
        settingsPanel.add(apply);

        // Add the main panel to the frame
        add(settingsPanel, BorderLayout.CENTER);

        // Connect button to the event handler inner class
        apply.addActionListener(new applyEvent());
    }

    /**
     * Inner class that handles the event when the user presses the "Apply Settings" button.
     * Reads current settings from the UI, validates input, saves the config file, refreshes the main UI,
     * and provides feedback via log text area.
     */
    public class applyEvent implements ActionListener {
        // Override the actionPerformed method to define the behavior when the event occurs
        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                // Construct a 2D array of settings values to be saved
                String[][] values = {
                        {"volume", volumeText.getText()},
                        {"symbols", sym},
                        {"sort", String.valueOf(sortCheckBox.isSelected())},
                        {"key", keyText.getText()},
                        {"realtime", String.valueOf(realtimeBox.isSelected())},
                        {"algo", String.valueOf(algoAggressivenessText.getText())},
                        {"candle", String.valueOf(candleBox.isSelected())},
                        {"T212", T212textField.getText()},
                        {"push", pushCutTextField.getText()}
                };

                // Save updated settings to the config XML file
                configHandler.saveConfig(values);

                // If candle chart mode changed, update chart and UI accordingly
                if (mainUI.useCandles != candleBox.isSelected()) {
                    mainUI.useCandles = candleBox.isSelected();
                    refreshChartType(true); // Redraw chart with new chart type
                }

                // Log the successful save operation
                logTextArea.append("Data saved successfully to config\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                // Reload updated config into the main application and refresh UI
                mainUI.loadConfig();
                mainUI.refreshAllComponents(gui.getContentPane());

                // Log the reload event
                logTextArea.append("Config updated and re-loaded\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                // Hide the settings dialog
                setVisible(false);

            } catch (Exception x) {
                // Handle any data entry or parsing errors
                x.printStackTrace();
                infos.setForeground(Color.RED); // Visual feedback (error in red)
                infos.setText("Wrong input in the text fields"); // Inform the user
            }
        }
    }
}