package org.crecker;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Settings_handler extends JFrame {
    public static JPanel settingsPanel;
    static JLabel volume, hype, infos, sort_label, keyLabel;
    static JTextField volume_text, hype_text, key_text;
    static JCheckBox sort_checkBox;
    int vol;
    float hyp;
    String sym, key;
    boolean sort;

    public Settings_handler(int vol, float hyp, String sym, boolean sort, String key) {
        setLayout(new BorderLayout(10, 10));
        this.vol = vol;
        this.hyp = hyp;
        this.sym = sym;
        this.sort = sort;
        this.key = key;

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
        volume_text = new JTextField(String.valueOf(vol), 15);
        volume_text.setAlignmentX(Component.LEFT_ALIGNMENT);

        hype = new JLabel("Hype:"); // Adding a label for clarity
        hype.setAlignmentX(Component.LEFT_ALIGNMENT);
        hype_text = new JTextField(String.valueOf(hyp), 5);
        hype_text.setAlignmentX(Component.LEFT_ALIGNMENT);

        sort_label = new JLabel("Sort hype entries");
        sort_label.setAlignmentX(Component.LEFT_ALIGNMENT);
        sort_checkBox = new JCheckBox();
        sort_checkBox.setSelected(sort);
        sort_checkBox.setAlignmentX(Component.LEFT_ALIGNMENT);

        keyLabel = new JLabel("API key:");
        keyLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        key_text = new JTextField(key);
        key_text.setAlignmentX(Component.LEFT_ALIGNMENT);

        // Add components to the settings panel with spacing
        settingsPanel.add(volume);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Space between description and input field
        settingsPanel.add(volume_text);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Space between fields
        settingsPanel.add(hype);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(hype_text);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(sort_label);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(sort_checkBox);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(keyLabel);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(key_text);
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
                // Retrieve and parse user input from text fields
                vol = Integer.parseInt(volume_text.getText());
                hyp = Float.parseFloat(hype_text.getText());

                String[][] values = {
                        {"volume", String.valueOf(vol)},
                        {"hype_strength", String.valueOf(hyp)},
                        {"symbols", sym},
                        {"sort", String.valueOf(sort_checkBox.isSelected())},
                        {"key", key_text.getText()}
                };

                config_handler.save_config(values);

                System.out.println("Data saved successfully to config");
                Main_UI.load_config();
                Main_UI.refresh(true, true, true, false);
                System.out.println("config updated and re-loaded");

                setVisible(false);
            } catch (Exception x) {
                // Handle any parsing errors that occur if the input is not in the expected format
                // Update the information label to inform the user of the error
                infos.setForeground(Color.RED); // Change the text color to red to indicate an error
                infos.setText("Wrong input in the text fields"); // Set the error message
                System.out.println("Wrong input in the text fields"); // Log the error message to the console
            }
        }
    }
}
