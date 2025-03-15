package org.crecker;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import static org.crecker.Main_UI.gui;
import static org.crecker.Main_UI.logTextArea;

public class Settings_handler extends JFrame {
    public static JPanel settingsPanel;
    static JLabel volume, infos, sortLabel, keyLabel, realtimeLabel;
    static JTextField volumeText, keyText;
    static JCheckBox sort_checkBox, realtimeBox;
    int vol;
    String sym, key;
    boolean sort, realtime;

    public Settings_handler(int vol, String sym, boolean sort, String key, boolean realtime) {
        setLayout(new BorderLayout(10, 10));
        this.vol = vol;
        this.sym = sym;
        this.sort = sort;
        this.key = key;
        this.realtime = realtime;

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

        volume = new JLabel("Volume:"); // Adding a label for clarity
        volume.setAlignmentX(Component.LEFT_ALIGNMENT);
        volumeText = new JTextField(String.valueOf(vol), 15);
        volumeText.setAlignmentX(Component.LEFT_ALIGNMENT);

        sortLabel = new JLabel("Sort hype entries");
        sortLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        sort_checkBox = new JCheckBox();
        sort_checkBox.setSelected(sort);
        sort_checkBox.setAlignmentX(Component.LEFT_ALIGNMENT);

        keyLabel = new JLabel("API key:");
        keyLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        keyText = new JTextField(key);
        keyText.setAlignmentX(Component.LEFT_ALIGNMENT);

        realtimeLabel = new JLabel("Chart realtime");
        realtimeLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        realtimeBox = new JCheckBox();
        realtimeBox.setSelected(realtime);
        realtimeBox.setAlignmentX(Component.LEFT_ALIGNMENT);

        // Add components to the settings panel with spacing
        settingsPanel.add(volume);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Space between description and input field
        settingsPanel.add(volumeText);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Space between fields
        settingsPanel.add(sortLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(sort_checkBox);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(keyLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(keyText);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(realtimeLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(realtimeBox);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));

        // Create and configure the "Apply Settings" button
        JButton apply = new JButton("Apply Settings");
        apply.setAlignmentX(Component.LEFT_ALIGNMENT); // Align the button to the left
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 20))); // Add space before the button
        settingsPanel.add(apply); // Add the button to the panel

        // Add the settings panel to the center of the frame
        add(settingsPanel, BorderLayout.CENTER);

        // Add action listener to the apply button, linking it to the event handling class
        apply.addActionListener(new apply_event());
    }

    // Inner class for handling the action of applying settings
    public class apply_event implements ActionListener {
        // Override the actionPerformed method to define the behavior when the event occurs
        @Override
        public void actionPerformed(ActionEvent e) {
            try {
                String[][] values = {
                        {"volume", volumeText.getText()},
                        {"symbols", sym},
                        {"sort", String.valueOf(sort_checkBox.isSelected())},
                        {"key", keyText.getText()},
                        {"realtime", String.valueOf(realtimeBox.isSelected())}
                };

                config_handler.save_config(values);

                logTextArea.append("Data saved successfully to config\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());

                Main_UI.load_config();
                Main_UI.refreshAllComponents(gui.getContentPane());

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