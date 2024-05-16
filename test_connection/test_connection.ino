#include "WiFi.h"
#include "ESPAsyncWebServer.h"

const char* ssid = "Testing WiFi";
const char* password = "123456789";

AsyncWebServer server(80);

String a;

void setup() {
  Serial.begin(115200);
  Serial.println();

  Serial.println("Setting AP (Access Point)...");
  WiFi.softAP(ssid, password);

  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);

  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    if (request->hasParam("data")) {
      String data = request->getParam("data")->value();
      request->send(200, "text/plain", "data recive: " + data);
      predicted = data;
      
    } else {
      request->send(400, "text/plain", "Bad Request: No message content provided");
    }
  });

  server.begin();
}

void loop() {
  // Print the received data
  Serial.println("Received data: " + predicted);
  delay(20);
}