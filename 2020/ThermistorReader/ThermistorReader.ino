
// thermistor pins
int thermistor0Pin = A0, thermistor1Pin = A1, thermistor2Pin = A2, thermistor3Pin = A3, thermistor4Pin = A4, thermistor5Pin = A5, thermistor6Pin = A6, thermistor7Pin = A7;
// thermistor coefficients`
float a = 0.003354016, b = 0.0002569850, c = 0.000002620131, d = 0.00000006383091;

void setup(void) {
 // start serial port
 Serial.begin(9600);
 Serial.print("Thermistor 1, ");
 Serial.print("Thermistor 2, ");
 Serial.print("Thermistor 3, ");
 Serial.print("Thermistor 4, ");
 Serial.print("Thermistor 5, ");
 Serial.print("Thermistor 6, ");
 Serial.print("Thermistor 7, ");
 Serial.print("Thermistor 8, ");
 Serial.println();
}

void loop(void) {
 float thermistor0Value = readTemperature(analogRead(thermistor0Pin));
 float thermistor1Value = readTemperature(analogRead(thermistor1Pin));
 float thermistor2Value = readTemperature(analogRead(thermistor2Pin));
 float thermistor3Value = readTemperature(analogRead(thermistor3Pin));
 float thermistor4Value = readTemperature(analogRead(thermistor4Pin));
 float thermistor5Value = readTemperature(analogRead(thermistor5Pin));
 float thermistor6Value = readTemperature(analogRead(thermistor6Pin));
 float thermistor7Value = readTemperature(analogRead(thermistor7Pin));
 
 Serial.print(thermistor0Value);
 Serial.print(", ");
 Serial.print(thermistor1Value);
 Serial.print(", ");
 Serial.print(thermistor2Value);
 Serial.print(", ");
 Serial.print(thermistor3Value);
 Serial.print(", ");
 Serial.print(thermistor4Value);
 Serial.print(", ");
 Serial.print(thermistor5Value);
 Serial.print(", ");
 Serial.print(thermistor6Value);
 Serial.print(", ");
 Serial.print(thermistor7Value);
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
