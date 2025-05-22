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

/**
 * <h1>configHandler</h1>
 * Utility class for managing the configuration file (config.xml).
 * Handles reading, writing, and initializing the application's configuration
 * settings using XML as the persistent storage format.
 * <p>
 * Provides methods to:
 * <ul>
 *     <li>Load configuration settings from the XML file.</li>
 *     <li>Save updated configuration settings to the XML file.</li>
 *     <li>Create a default configuration file with preset settings.</li>
 * </ul>
 * The configuration is stored as key-value pairs, supporting dynamic and extensible settings.
 */
public class configHandler {

    /**
     * Loads configuration settings from the config.xml file.
     * If the file does not exist or is corrupted, creates a default config and reloads.
     *
     * @return a 2D String array containing key-value pairs of config settings.
     */
    public static String[][] loadConfig() {
        try {
            // Construct the file path for the config.xml in the current working directory
            Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
            File inputFile = configPath.toFile();

            // Create the document builder and parse the XML file into a DOM Document object
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(inputFile);

            // Normalize the XML document to standardize structure (combines adjacent text nodes)
            doc.getDocumentElement().normalize();

            // Get the list of all child nodes under the <config> root element
            NodeList nodeList = doc.getDocumentElement().getChildNodes();

            // First, count only ELEMENT_NODEs to know how many config entries exist
            int numEntries = 0;
            for (int i = 0; i < nodeList.getLength(); i++) {
                if (nodeList.item(i).getNodeType() == Node.ELEMENT_NODE) {
                    numEntries++;
                }
            }

            // Prepare the result array for key-value pairs
            String[][] values = new String[numEntries][2];

            // Iterate again, this time extracting key (tag) and value (text) from each element
            int entryIndex = 0;
            for (int i = 0; i < nodeList.getLength(); i++) {
                Node node = nodeList.item(i);
                if (node.getNodeType() == Node.ELEMENT_NODE) {
                    values[entryIndex][0] = node.getNodeName();      // The element name acts as the key
                    values[entryIndex][1] = node.getTextContent();   // Text content as the value
                    entryIndex++;
                }
            }

            // Return the 2D array containing all the configuration entries
            return values;

        } catch (Exception e) {
            // If parsing fails (file missing, corrupted, or invalid), print the error and create a new config
            e.printStackTrace();
            createConfig();
            // Retry loading after creating the default config file
            return loadConfig();
        }
    }

    /**
     * Saves the provided key-value settings into config.xml, replacing its contents.
     *
     * @param values 2D String array containing [key][value] pairs to write to the config file.
     * @throws RuntimeException if any IO or XML error occurs during save.
     */
    public static void saveConfig(String[][] values) {
        try {
            // Prepare to build a new XML Document
            DocumentBuilderFactory documentFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder documentBuilder = documentFactory.newDocumentBuilder();

            // Create a new Document, which is the base representation of an XML document
            Document document = documentBuilder.newDocument();

            // Create and append the <config> root element
            Element rootElement = document.createElement("config");
            document.appendChild(rootElement);

            // Add each configuration entry as a child element under <config>
            for (String[] entry : values) {
                String name = entry[0];    // Key/tag name
                String content = entry[1]; // Associated value

                Element element = document.createElement(name); // Create element for the key
                element.appendChild(document.createTextNode(content)); // Set the text content
                rootElement.appendChild(element); // Attach to root
            }

            // Prepare a Transformer to write the DOM Document back to an XML file with pretty print
            TransformerFactory transformerFactory = TransformerFactory.newInstance();
            Transformer transformer = transformerFactory.newTransformer();
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");

            // Set up the file path for output
            Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
            File configFile = configPath.toFile();

            // Write the DOM Document to the file as XML
            DOMSource domSource = new DOMSource(document);
            StreamResult streamResult = new StreamResult(configFile);

            // Perform the transformation from the Document to an XML file
            transformer.transform(domSource, streamResult);

            // Print a success message indicating the XML file was saved successfully
            System.out.println("Config file saved successfully!");

        } catch (Exception e) {
            // Print the error for debugging and throw a runtime exception to propagate failure
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    /**
     * Creates a new default configuration file (config.xml) in the current directory.
     * Overwrites any existing config.xml. Called automatically if the config file is missing/corrupted.
     * <p>
     * Default values are set for all expected keys.
     *
     * @throws RuntimeException if file creation or XML operations fail.
     */
    public static void createConfig() {
        try {
            // Create the XML Document and <config> root
            DocumentBuilderFactory builderFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder builder = builderFactory.newDocumentBuilder();
            Document doc = builder.newDocument();

            // This is the main container for all the configuration elements that will be added later
            Element root = doc.createElement("config");
            doc.appendChild(root);

            // Create and add all required config settings with default values:

            // Example: Default volume setting
            Element volume = doc.createElement("volume");
            volume.appendChild(doc.createTextNode("40000"));
            root.appendChild(volume);

            // Example: Symbols list with associated colours (as a string)
            Element symbols = doc.createElement("symbols");
            symbols.appendChild(doc.createTextNode(
                    "[MSFT,java.awt.Color[r=221,g=160,b=221]]," +
                            "[NVDA,java.awt.Color[r=102,g=205,b=170]]," +
                            "[GOOGL,java.awt.Color[r=255,g=182,b=193]]," +
                            "[AAPL,java.awt.Color[r=135,g=206,b=250]]," +
                            "[TSLA,java.awt.Color[r=240,g=230,b=140]]"
            ));
            root.appendChild(symbols);

            // Example: Sorting setting
            Element sort = doc.createElement("sort");
            sort.appendChild(doc.createTextNode("false"));
            root.appendChild(sort);

            // Example: API key setting (public/free API)
            Element key = doc.createElement("key");
            key.appendChild(doc.createTextNode(""));
            root.appendChild(key);

            // Example: Real-time data fetch toggle
            Element realtime = doc.createElement("realtime");
            realtime.appendChild(doc.createTextNode("false"));
            root.appendChild(realtime);

            // Example: Algorithm version used for data processing
            Element algo = doc.createElement("algo");
            algo.appendChild(doc.createTextNode("1.0"));
            root.appendChild(algo);

            // Example: Whether to use candlestick charts or not
            Element useCandles = doc.createElement("candle");
            useCandles.appendChild(doc.createTextNode("false"));
            root.appendChild(useCandles);

            // Example: Trading212 API key
            Element Trading212Key = doc.createElement("T212");
            Trading212Key.appendChild(doc.createTextNode(""));
            root.appendChild(Trading212Key);

            // Example: PushCut notification endPoint
            Element pushCutUrl = doc.createElement("push");
            pushCutUrl.appendChild(doc.createTextNode(""));
            root.appendChild(pushCutUrl);

            // Example: Greed Mode activation
            Element greed = doc.createElement("greed");
            greed.appendChild(doc.createTextNode("false"));
            root.appendChild(greed);

            // Example: Market selection
            Element market = doc.createElement("market");
            market.appendChild(doc.createTextNode("allSymbols"));
            root.appendChild(market);

            // Add any additional default settings below as needed for the application

            // Prepare to write the DOM Document to XML file with indentation (pretty print)
            TransformerFactory transformerFactory = TransformerFactory.newInstance();
            Transformer transformer = transformerFactory.newTransformer();

            // Configure the Transformer for making the output readable
            transformer.setOutputProperty(OutputKeys.INDENT, "yes");
            transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "4");

            // Specify the destination file path
            Path configPath = Paths.get(System.getProperty("user.dir"), "config.xml");
            File configFile = configPath.toFile();

            // Write the Document to the file
            DOMSource domSource = new DOMSource(doc);
            StreamResult result = new StreamResult(configFile);

            // Transform the XML Document into an XML file on disk
            transformer.transform(domSource, result);

            System.out.println("Config file created successfully!");

        } catch (Exception e) {
            // Print and rethrow the error as a runtime exception for debugging/handling upstream
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }
}