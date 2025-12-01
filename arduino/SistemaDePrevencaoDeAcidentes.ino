#define LED_PIN 13

char ultimoEstado = '0';

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char dado = Serial.read();

    if (dado != ultimoEstado) {
      ultimoEstado = dado;  // Atualiza estado somente quando mudar

      if (dado == '1') {
        digitalWrite(LED_PIN, HIGH);
        Serial.println("ALERTA: perigo detectado (LED ON)");
      } 
      else if (dado == '0') {
        digitalWrite(LED_PIN, LOW);
        Serial.println("Normal (LED OFF)");
      }
    }
  }
}

