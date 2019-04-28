/* Pranav Addepalli


*/

// thermistor pins
int thermistor0Pin = A0, thermistor1Pin = A1, thermistor2Pin = A2, thermistor3Pin = A3;
// thermistor coefficients`
float a = 0.003354016, b = 0.0002569850, c = 0.000002620131, d = 0.00000006383091;

void setup(void) {
  // start serial port
  Serial.begin(9600);
  Serial.print("Thermistor 1,");
  Serial.println("Thermistor 2");
  
   
}

void loop(void) {
  float thermistor0Value = readTemperature(analogRead(thermistor0Pin));
   float thermistor1Value = readTemperature(analogRead(thermistor1Pin));
 

  Serial.print(thermistor0Value);
  Serial.print(",");
  /*Serial.print(thermistor1Value);*/
  Serial.println();
  delay(1000);

}

float readTemperature(double reading) {
  float temperature;
  // convert the analog reading into a resistance
  float resistance = (10000.0 * ((1024.0 / reading - 1)));
  // temporary variable
  float z = log(resistance / 20000);
  temperature = (1 / (a + (b * z) + (c * z * z) + (d * z * z * z))) - 273.15;

  return temperature;
}
