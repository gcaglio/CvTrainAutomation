# Overview

This project use OpenCV and EmguCV to implements a computer vision solution to recognize specific patterns and trigger workflows to invoke external services. The system supports multiple USB webcams and can identify multiple patterns in video streams. Each pattern, defined in the configuration file, also specifies a multi-step workflow to execute. Each step can involve:
- Invoking an HTTPS endpoint
- Publishing strings to MQTTs topics
- Waiting for a specified time

This allows different sequences for each pattern, making the software more versatile.

# Configuration - Webcam

The `config.json` file contains an array of JSON objects like the following:

```json
{
  "Id": "USB_1",
  "StreamIndex": 2,
  "Width": 800,
  "Height": 600,
  "Fps": 15,
  "Focus": 1
}
```

It is possible to define the resolution, frames per second, and focus (typically 1 = near, 100 = far) for each webcam.

**Note:** DirectShow does not support 2 webcams on the same USB bus, based on available documentation and performed tests. You need 2 separate USB controllers. On Windows, you can use the MMC "Device Management" to check your PC hardware.

The most important parameter for identifying the correct webcam is:

```json
"StreamIndex": X
```

If you don’t know the stream index of the desired webcam, you can enter a random number (e.g., 999) and start the program. Upon startup, it will display a list of all webcams with their corresponding names and StreamIndex, which you can use to complete the configuration.

# Configuration - Pattern

The `config_pattern.json` file contains the definition of patterns to search for in the video streams and the workflow that the software will execute when it finds the specific pattern.

Example:

```json
{
  "Id": 1,
  "Description": "sample pattern",
  "Path": "C:\\Temp\\sample_pattern.png",
  "Cooldown": 30000,
  "Hooks": [
    {
      "Type": "sleep",
      "Endpoint": "",
      "Payload": "3000"
    },
    {
      "Type": "mqtt",
      "Endpoint": "mqtts://9a9a9a9a9a9a9a9a9a9a9a9a.s2.eu.hivemq.cloud:8883",
      "Username": "XXXXXXXX",
      "Password": "XXXXXXXX01",
      "Topic": "/mytopic",
      "Description": "go back 3 sec",
      "Payload": "GOBACK_3S"
    }
  ]
}
```

This JSON defines:
- A sample pattern in the image located at `c:\temp\sample_pattern.png`
- A cooldown interval (ignore any other frame in the video stream) for 30 seconds = 30000 milliseconds
- Two actions to perform when the pattern is found:
  - Sleep for 3 seconds
  - Publish "GOBACK_3S" to the `/mytopic` on the MQTTs broker defined by the `Endpoint`, authenticated with `Username` and `Password`

# Samples

The project contains several folders with examples of architecture and implementation.

**Note:** Some examples, topics, and implementations reference ESP8266 devices running my CloudBrick sketch.

Example folders:
- **sample:** Contains a JSON file and an image showing binary configuration and webcam positioning. By reading the JSON, you can understand, via the description fields, the actions being performed.
- **sample_doc:** A draw.io file and PNG with an HLD (high-level design) of the implemented architecture, including 4 ESP8266 devices, the webcam, this software, the free HiveMQ broker, and the web interface for manual interventions.
- **pattern:** Examples of successfully used images. These include geometric shapes, a "plate" made from some Lego bricks, and the detailed back of a driver's cabin.

# Tuning

It’s recommended to run the software in a well-lit environment.

It's recommended to capture/save images directly from the in-place webcam.  
You can use an image editor to crop and resize the image to have a better sample.  
Hint : selecting small characteristic areas increase the confidence percentage (eg: you may consider to use a sticker with a simple geometric shape)  
If you opt to create "plates", use distinct colors and patterns to facilitate recognition and increase confidence percentages.

Frame analysis requires CPU power. If you notice delays in recognition, try lowering the webcam’s resolution or frame rate.  
You may consider to reduce the pattern image resolution or size.

As seen in the demo video, it may happen that recognition does not occur on the first webcam pass, or even the second, but the third or fourth. This is normal: analysis requires a confidence level >60% to trigger the workflow. If objects move quickly or conditions are unstable (e.g., shaking, poor lighting), recognition won’t have the necessary confidence to "make a decision."

During the initial phases of defining and testing the workflow, it’s recommended to keep an MQTT interface or client handy for manual interventions, to prevent potential crashes or errors.
