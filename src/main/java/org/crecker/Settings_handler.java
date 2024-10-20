package org.crecker;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class Settings_handler extends JFrame {

    // JTextField components for user input
    JTextField resolution; // Text field for the user to input the desired resolution
    JTextField epochs; // Text field for the user to input the number of epochs
    JTextField batch; // Text field for the user to input the batch size
    JTextField display_scale; // Text field for the user to input the data visualization scale

    // Descriptive labels to guide the user in the GUI
    JLabel resolution_desc; // Label describing the resolution input field
    JLabel epochs_desc; // Label describing the epochs input field
    JLabel batch_desc; // Label describing the batch size input field
    JLabel display_scale_desc; // Label describing the display scale input field
    JLabel infos; // Label for displaying informational messages or error messages

    // Constructor for the model_param class to set up the user interface for model parameter configuration
    public Settings_handler() {
        // Set the layout manager for the frame to BorderLayout with specified horizontal and vertical gaps
        setLayout(new BorderLayout(10, 10));

        // Create a panel to hold the settings components
        JPanel settingsPanel = new JPanel();
        // Set the layout of the panel to BoxLayout, which arranges components vertically
        settingsPanel.setLayout(new BoxLayout(settingsPanel, BoxLayout.Y_AXIS));
        // Add a titled border to the panel for clarity
        settingsPanel.setBorder(BorderFactory.createTitledBorder("Model Parameters for training models"));

        // Informational label to guide the user
        infos = new JLabel("Select your settings and then press apply");
        // Add the label to the panel
        settingsPanel.add(infos);
        // Add space between the label and the next component
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 10)));

        // Create and add components for the resolution setting
        resolution_desc = new JLabel("Picture size - to x * x pixel the images will be downscaled first before processing (larger value more information but longer processing time)");
        resolution = new JTextField(String.valueOf(res), 4); // Create a text field with the current resolution as its initial value

        // Create and add components for the epochs setting
        epochs_desc = new JLabel("Epochs - How many training rounds");
        epochs = new JTextField(String.valueOf(epo), 4); // Create a text field for the number of epochs

        // Create and add components for the batch size setting
        batch_desc = new JLabel("Batch size - how many images should be used for training at once");
        batch = new JTextField(String.valueOf(bat), 4); // Create a text field for batch size

        // Create and add components for the display scale setting
        display_scale_desc = new JLabel("Data visualisation scale for the displaying and analysis of training");
        display_scale = new JTextField(String.valueOf(lea), 10); // Create a text field for display scale

        // Add components to the settings panel with spacing
        settingsPanel.add(resolution_desc);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Space between description and input field
        settingsPanel.add(resolution);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Space between fields
        settingsPanel.add(epochs_desc);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(epochs);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(batch_desc);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(batch);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(display_scale_desc);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5)));
        settingsPanel.add(display_scale);
        settingsPanel.add(Box.createRigidArea(new Dimension(0, 5))); // Add space before the button

        // Create and configure the "Apply Settings" button
        JButton apply = new JButton("Apply Settings");
        apply.setAlignmentX(Component.CENTER_ALIGNMENT); // Center the button in the panel
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
                // The user inputs are expected to be in specific formats (int or float)
                int res = Integer.parseInt(resolution.getText());  // Parse resolution as an integer
                int epo = Integer.parseInt(epochs.getText());      // Parse epochs as an integer
                int bat = Integer.parseInt(batch.getText());        // Parse batch size as an integer
                float lea = Float.parseFloat(display_scale.getText()); // Parse learning rate as a float

                // Create an instance of the main UI class to call the method that saves and reloads the configuration
                Main_UI mainUI = new Main_UI();
                // Save the configuration and reload it with the new settings
                mainUI.save_reload_config(res, epo, bat, lea, pic, ten);

                // Close the settings window by setting its visibility to false
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
