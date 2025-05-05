package org.crecker;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

public class configHandler {
    public static String[][] loadConfig() {
        // Method for loading the configuration data from the config.xml file
        // It returns a 2D array where [index][0] is the key and [index][1] is the value
        try {
            // Create a new File instance pointing to the config.xml file
            Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
            File inputFile = configPath.toFile();

            // Create a DocumentBuilderFactory instance which is used to create a DocumentBuilder
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            // Create a DocumentBuilder from the factory to parse XML files
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();

            // Parse the config.xml file and create an in-memory representation of the XML document
            org.w3c.dom.Document doc = dBuilder.parse(inputFile);

            // Normalize the XML structure (helpful in combining adjacent text nodes)
            doc.getDocumentElement().normalize();

            // Get all child nodes of the <config> root element
            NodeList nodeList = doc.getDocumentElement().getChildNodes();

            // Count the number of valid element nodes (ignoring other node types like text or comments)
            int numEntries = 0;
            for (int i = 0; i < nodeList.getLength(); i++) {
                if (nodeList.item(i).getNodeType() == Node.ELEMENT_NODE) {
                    numEntries++; // count the element nodes so we can create the array with the correct size
                }
            }

            // Initialize a 2D array to store the keys and values from the XML file
            String[][] values = new String[numEntries][2]; // Each entry will hold the key-value pair

            // Load data from the XML into the array
            int entryIndex = 0;
            for (int i = 0; i < nodeList.getLength(); i++) {
                Node node = nodeList.item(i); // Get the current node
                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    // If the node is an element, store its name (key) and text content (value)
                    values[entryIndex][0] = node.getNodeName(); // Store the name of the element as key
                    values[entryIndex][1] = node.getTextContent(); // Store the text content of the element as value
                    entryIndex++; // Move to the next array index
                }
            }

            // Return the 2D array containing all the configuration entries
            return values;

        } catch (Exception e) {
            // Handle any exceptions that occur during XML processing
            e.printStackTrace();

            // If an error occurs, it's because the config file does not exist or is corrupted.
            // Call createConfig() to generate a new default configuration file
            createConfig();

            // Reload the configuration using the newly created config file
            return loadConfig();
        }
    }

    public static void saveConfig(String[][] values) {
        try {
            // Create an instance of DocumentBuilderFactory, which provides a way to obtain a DocumentBuilder
            // This factory enables the creation of XML documents
            DocumentBuilderFactory documentFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder documentBuilder = documentFactory.newDocumentBuilder();

            // Create a new Document, which is the base representation of an XML document
            Document document = documentBuilder.newDocument();

            // Create the root element of the XML (in this case, <config>) and append it to the document
            // This will serve as the container for all configuration settings
            Element rootElement = document.createElement("config");
            document.appendChild(rootElement);

            // Iterate through the provided values to create child elements for each configuration setting
            // Each setting is represented as a pair where the first element is the tag name, and the second is its value
            for (String[] entry : values) {
                String name = entry[0];  // Get the element name from the provided values
                String content = entry[1];  // Get the content/value for the element

                // Create a new element for each configuration entry using the specified name
                Element element = document.createElement(name);
                // Set the content of the element to the value provided
                element.appendChild(document.createTextNode(content));
                // Append this new element to the root element of the XML document
                rootElement.appendChild(element);
            }

            // Set up a TransformerFactory and Transformer, which will be used to convert the XML Document into a file
            TransformerFactory transformerFactory = TransformerFactory.newInstance();
            Transformer transformer = transformerFactory.newTransformer();

            // Configure the Transformer to output the XML in a readable (indented) format
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");

            // Define the source (the document we created) and the destination (the output file)
            DOMSource domSource = new DOMSource(document);

            // Get the path dynamically
            Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
            File configFile = configPath.toFile();

            // Use the file in StreamResult
            StreamResult streamResult = new StreamResult(configFile);

            // Perform the transformation from the Document to an XML file
            transformer.transform(domSource, streamResult);

            // Print a success message indicating the XML file was saved successfully
            System.out.println("Config file saved successfully!");

        } catch (Exception e) {
            // Handle any exceptions that occur during the process by throwing a RuntimeException
            // This ensures any errors are clearly reported back to the user
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    public static void createConfig() { //method for creating a new config file
        try {
            // Create a DocumentBuilderFactory instance to produce a DocumentBuilder
            // The DocumentBuilderFactory provides a way to obtain a DocumentBuilder instance
            DocumentBuilderFactory builderFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = builderFactory.newDocumentBuilder();

            // Create a new Document, which is a blank XML document
            Document doc = builder.newDocument();

            // Create the root element of the XML file, called <config>, and append it to the document
            // This is the main container for all the configuration elements that will be added later
            Element root = doc.createElement("config");
            doc.appendChild(root);

            Element volume = doc.createElement("volume");
            volume.appendChild(doc.createTextNode("1000"));
            root.appendChild(volume);

            Element symbols = doc.createElement("symbols");
            symbols.appendChild(doc.createTextNode("[MSFT,java.awt.Color[r=221,g=160,b=221]],[NVDA,java.awt.Color[r=102,g=205,b=170]],[GOOGL,java.awt.Color[r=255,g=182,b=193]],[AAPL,java.awt.Color[r=135,g=206,b=250]],[TSLA,java.awt.Color[r=240,g=230,b=140]]"));
            root.appendChild(symbols);

            Element sort = doc.createElement("sort");
            sort.appendChild(doc.createTextNode("false"));
            root.appendChild(sort);

            Element key = doc.createElement("key");
            key.appendChild(doc.createTextNode("0988PSIKXZ50IP2T")); // Free API key
            root.appendChild(key);

            Element realtime = doc.createElement("realtime");
            realtime.appendChild(doc.createTextNode("false"));
            root.appendChild(realtime);

            Element algo = doc.createElement("algo");
            algo.appendChild(doc.createTextNode("1.0"));
            root.appendChild(algo);

            Element useCandles = doc.createElement("candle");
            useCandles.appendChild(doc.createTextNode("false"));
            root.appendChild(useCandles);

            // Additional elements can be added here as needed (by creating new tags and appending to root)

            // Use TransformerFactory to create a Transformer instance for converting the Document to XML format
            TransformerFactory transformerFactory = TransformerFactory.newInstance();
            Transformer transformer = transformerFactory.newTransformer();

            // Configure the Transformer for making the output readable
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");
            transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4"); // Setting the indent to 4 spaces for better formatting

            // Set up the input source (the document) and output target (the config.xml file)
            DOMSource domSource = new DOMSource(doc);
            // Get the path dynamically
            Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
            File configFile = configPath.toFile();

            // Use the file in StreamResult
            StreamResult result = new StreamResult(configFile);

            // Transform the XML Document into an XML file on disk
            transformer.transform(domSource, result);

            // Print a message indicating that the XML configuration file was created successfully
            System.out.println("Config file created successfully!");

        } catch (Exception e) {
            // Handle any exception that may occur by throwing a RuntimeException
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }
}