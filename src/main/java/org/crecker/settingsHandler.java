package org.crecker;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import static org.crecker.mainUI.gui;
import static org.crecker.mainUI.logTextArea;

public class settingsHandler extends JFrame {
    public static JPanel settingsPanel;
    static JLabel volume, infos, sortLabel, keyLabel, realtimeLabel, algoLabel;
    static JTextField volumeText, keyText, algoAggressivenessText;
    static JCheckBox sortCheckBox, realtimeBox;
    int vol;
    float aggressiveness;
    String sym, key;
    boolean sort, realtime;

    public settingsHandler(int vol, String sym, boolean sort, String key, boolean realtime, float aggressiveness) {
        setLayout(new BorderLayout(10, 10));
        this.vol = vol;
        this.sym = sym;
        this.sort = sort;
        this.key = key;
        this.realtime = realtime;
        this.aggressiveness = aggressiveness;

        // Create a panel to hold the settings components
        settingsPanel = new JPanel();
        // Set the layout of the panel to BoxLayout, which arranges components vertically
        settingsPanel.setLayout(new BoxLayout(settingsPanel, BoxLayout.Y_AXIS));
        // Add a titled border to the panel for clarity
        settingsPanel.setBorder(BorderFactory.createTitledBorder("Settings for Stock management"));

        // Informational label to guide the user
        infos = new JLabel("Select your settings and then press apply");
        // Add the label to the panel and set its alignment to the left
        infos.setAlignmentX(Component.LEFT_ALIGNMENT);
        settingsPanel.add(infos);
        // Add space between the label and the next component
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 10)));

        volume = new JLabel("Volume in Euro/USD:"); // Adding a label for clarity
        volume.setAlignmentX(Component.LEFT_ALIGNMENT);
        volumeText = new JTextField(String.valueOf(vol), 15);
        volumeText.setAlignmentX(Component.LEFT_ALIGNMENT);

        sortLabel = new JLabel("Sort hype entries by change automatically");
        sortLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        sortCheckBox = new JCheckBox();
        sortCheckBox.setSelected(sort);
        sortCheckBox.setAlignmentX(Component.LEFT_ALIGNMENT);

        keyLabel = new JLabel("API key (Premium Key for Hype Mode required):");
        keyLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        keyText = new JTextField(key);
        keyText.setAlignmentX(Component.LEFT_ALIGNMENT);

        realtimeLabel = new JLabel("Chart realtime updates (per second)");
        realtimeLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        realtimeBox = new JCheckBox();
        realtimeBox.setSelected(realtime);
        realtimeBox.setAlignmentX(Component.LEFT_ALIGNMENT);

        algoLabel = new JLabel("<html>Hype aggressiveness<br>(lower number for more but less precise entries, vice versa)</html>");
        algoLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        algoAggressivenessText = new JTextField(String.valueOf(aggressiveness));
        algoAggressivenessText.setAlignmentX(Component.LEFT_ALIGNMENT);

        // Add components to the settings panel with spacing
        settingsPanel.add(volume);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Space between description and input field
        settingsPanel.add(volumeText);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Space between fields
        settingsPanel.add(sortLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(sortCheckBox);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(keyLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(keyText);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(realtimeLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(realtimeBox);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(algoLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(algoAggressivenessText);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));

        // Create and configure the "Apply Settings" button
        JButton apply = new JButton("Apply Settings");
        apply.setAlignmentX(Component.LEFT_ALIGNMENT); // Align the button to the left
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space before the button
        settingsPanel.add(apply); // Add the button to the panel

        // Add the settings panel to the center of the frame
        add(settingsPanel, BorderLayout.CENTER);

        // Add action listener to the apply button, linking it to the event handling class
        apply.addActionListener(new applyEvent());
    }

    // Inner class for handling the action of applying settings
    public class applyEvent implements ActionListener {
        // Override the actionPerformed method to define the behavior when the event occurs
        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                String[][] values = {
                        {"volume", volumeText.getText()},
                        {"symbols", sym},
                        {"sort", String.valueOf(sortCheckBox.isSelected())},
                        {"key", keyText.getText()},
                        {"realtime", String.valueOf(realtimeBox.isSelected())},
                        {"algo", String.valueOf(algoAggressivenessText.getText())}
                };

                configHandler.saveConfig(values);

                logTextArea.append("Data saved successfully to config\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                mainUI.loadConfig();
                mainUI.refreshAllComponents(gui.getContentPane());

                logTextArea.append("Config updated and re-loaded\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                setVisible(false);
            } catch (Exception x) {
                x.printStackTrace();
                // Handle any parsing errors that occur if the input is not in the expected format
                // Update the information label to inform the user of the error
                infos.setForeground(Color.RED); // Change the text color to red to indicate an error
                infos.setText("Wrong input in the text fields"); // Set the error message
            }
        }
    }
}