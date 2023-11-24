# Entregable 2
Hola, bienvenido. 
Esta es la documentación del proyecto final!!
Estoy trabajando con [Silvana Olivares](https://github.com/kquita) y [val3ntiina](https://github.com/val3ntiina)



# Bender Identifier
Permite que el maestro Arduino identifique tu elemento.



![Captura 2](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/d57546f4-74c6-4636-b5b5-2dfc1521bca7)

[Noviembre, 2023]

Estudiantes: [Sofia Alarcon](https://github.com/sofantasyy), [Silvana Olivares](https://github.com/kquita) y [Valentina Abarzua](https://github.com/val3ntiina)

Profesores: [Aaron Montoya](https://github.com/montoyamoraga)

Ayudante : [Ignacio Passalacqua](https://github.com/ipassala)

Ramo: Inteligencia artificial (AUDIV027-1)
_____
## Acercamientos previos
En nuestra primera aproximación con Arduino, nos llamó la atención el ejercicio "Fruit identification using Arduino and TensorFlow", que en ese entonces modificamos la base de datos para ingresar nuestro propio registro cromático y su asociación con un objeto distinto al original.

![Captura 1](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/ba33585b-9195-4943-a2a8-7b7b2927009c)

También tuvimos la posibilidad de utilizar Micro_Speech de Harvard_TinyMLx. Inteligencia que permitía realizar asociaciones entre las palabras "Yes", "No" y "Unknown" escuchadas, una reapusta del encendido de la luz LED del microcontrolador en colores Verde, Rojo y Azul respectivamente.


![WhatsApp Image 2023-11-10 at 15 50 28 (1)](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/04a15028-5431-4bbc-9b49-b9f2c9e157f9)

A partir de estos dos ejemplos estudiados decidimos mezclar ambas inteligencias para que sea posible clasificar un elemento según sus cualidades cromáticas, desde los elementos Agua, Tierra, Fuego y Aire. Adicional a la respuesta de texto en la pantalla con su respectiva probabilidad de confianza, más una respuesta simultánea en el microcontrolador con el encendidido del LED con los colores Azul, Verde, Rojo y Amarillo según corresponda.
## Registro del proceso 
### PASO 1
Hicimos registro de cada elemento, asociado a un color particular. Para el elemento Fuego=Rojo, Agua=Azul, Tierra=Verde y Viento=Amarillo. Estos datos fueron ingresados en Object_color_capture.ino

![WhatsApp Image 2023-11-10 at 17 07 05](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/ba57dd30-d2e6-4947-a0a2-19f359f121fa)

![WhatsApp Image 2023-11-10 at 17 07 07](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/b7243c1e-58c7-43ef-8b3c-77fe208de28f)


### PASO 2
La información recopilada en Serial Monitor, se copia y luego e pega en los archivos CSV que se deben abrir con brackets. Esto se hace uno a uno con cada muestra.

![WhatsApp Image 2023-11-10 at 17 11 56](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/a9a7eaad-dae6-4554-b5d0-127b48c5a154)


### PASO 3
Una vez recipilados los archivos CSV con el registro cromático de cada elemento(Fuego, Agua, Tierra y viento), debemos subir estas carpetas al documento de Google Colab: https://colab.research.google.com/github/arduino/ArduinoTensorFlowLiteTutorials/blob/master/FruitToEmoji/FruitToEmoji.ipynb en la carpeta de archivos y también eliminar la celda completa de Run Whit Test Data. Ya que esta celda solo permite trabajar con 3 variables. IMPORTANTE: NO CARGAR LOS ARCHIVOS EN SAMPLE DATA COMO LA IMAGEN SIGUENTE.

![WhatsApp Image 2023-11-10 at 17 32 41](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/1f0f61a7-ffed-4d4f-ac17-3734d9138751)

Así debe verse:)

![WhatsApp Image 2023-11-10 at 17 51 47](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/e43a508b-88a0-45b9-9305-4b7fe1aa12a2)


### PASO 3
Luego debe hacer correr todo el entorno de ejecución, refrescar las carpetas(1) y posteriormente descargar el Model h(2)

![InkedInkedWhatsApp Image 2023-11-10 at 18 00 31](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/5194d645-13ac-4e10-ae63-647a4ed012cd)


### PASO 4
Desde el archivo original de Object_color_classify.ino, se elimina el Model h existente y lo remplazamos por el que obtuvimos desde Google Colab. Ahora es posible clasificar elementos según la base de datos cromática ingresada.

![7bc5a488-afe6-4a85-b01c-802c43e85bc1](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/71de6014-44ce-4c1f-93b4-3cb4a6b49af4)


### PASO 5
Ahora rescatamos el código desde Micro_Speech de Harvard_TinyMLx. relacionado con el encendido de la luz LED del microcontrolador. Esto con la finalidad de relacionar la respuesta de clasificación(Agua, Aire, Fuego o Tierra) con una respuesta lumínica adicional si el elemento clasificado posee una probabilidad de confianza mayor al 50%. Agua= Luz Azul, Fuego= Luz Roja, Tierra= Luz Verde y Aire= Luz Blanca.


A pesar de haber ingresado el código para que exista una respuesta LED adicional a la que se presenta en la pantalla, al probar el clasificador, ninguna luz se enciende.



## Código
```


// inicio referencia
// https://github.com/arduino/ArduinoTensorFlowLiteTutorials/blob/master/FruitToEmoji/sketches/object_color_classify/object_color_classify.ino

// Arduino_TensorFlowLite - Version: 0.alpha.precompiled
#include <TensorFlowLite.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <Arduino_APDS9960.h>
#include "model.h"

// global variables used for TensorFlow Lite (Micro)
// Variables globales usadas por TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
//Puedes remover todas las opciones TFLM innecesarias
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
//tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
//Creamos un buffer de memoria para TFLM (se ajusta al modelo que se usa)
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize];

// array to map gesture index to a name
// este es un mapa de indice de clases
// Cambiamos las frutas por elementos y agrgamos una clase nueva
const char* CLASSES[] = {
  "Agua",   // u8"\U0001F34E", // Apple
  "Aire",   // u8"\U0001F34C", // Banana
  "Fuego",  // u8"\U0001F34A"  // Orange
  "Tierra"
};

#define NUM_CLASSES (sizeof(CLASSES) / sizeof(CLASSES[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);
  
  //Si hay un error saldra impreso en el monitor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
    while (1);
  }
  
  if (!TFL.begin(model, NUM_CLASSES)) {
    Serial.println("Error initializing TensorFlow Lite.");
    while (1);
  }

  pinMode(LEDR, OUTPUT);
  pinMode(LEDB, OUTPUT);
  pinMode(LEDG, OUTPUT);
}

//Ahora para la sigueiente parte nuestra referencia es 
//https://docs.arduino.cc/tutorials/nano-33-ble-sense/rgb-sensor
void loop() {
  int r, g, b, p, c;
  float sum;

  while (!APDS.colorAvailable() || !APDS.proximityAvailable()) {}

  APDS.readColor(r, g, b, c);
  p = APDS.readProximity();
  sum = r + g + b;

  if (p == 0 && c > 10 && sum > 0) {
    float redRatio = r / sum;
    float greenRatio = g / sum;
    float blueRatio = b / sum;

    TfLiteStatus invokeStatus = TFL.invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      while (1);
    }

 // Segun la clase con una probabilidad mayor al 50% 
 // se encendera una luz distinta vinculada a cada clase
    for (int i = 0; i < NUM_CLASSES; i++) {
      Serial.print(CLASSES[i]);
      Serial.print(" ");
      Serial.print(int(TFL.output()[i] * 100));
      Serial.print("%\n");

      if (TFL.output()[i] > 0.51) {
        if (strcmp(CLASSES[i], "Fuego") == 0) {
          digitalWrite(LEDR, LOW);  // Rojo para fuego
          digitalWrite(LEDG, HIGH);
          digitalWrite(LEDB, HIGH);
        } else if (strcmp(CLASSES[i], "Agua") == 0) {
          digitalWrite(LEDB, LOW);  // Azul para agua
          digitalWrite(LEDG, HIGH);
          digitalWrite(LEDR, HIGH);
        } else if (strcmp(CLASSES[i], "Tierra") == 0) {
          digitalWrite(LEDG, LOW);  // Verde para tierra
          digitalWrite(LEDR, HIGH);
          digitalWrite(LEDB, HIGH);
        } else if (strcmp(CLASSES[i], "Aire") == 0) {
          digitalWrite(LEDR, HIGH);  // Blanco para aire
          digitalWrite(LEDG, HIGH);
          digitalWrite(LEDB, HIGH);
        }
      }
    }

// Se imprime en el monitor la probabilidad luego de que se 
//activen los sensores de proximidad
    Serial.println();

    while (!APDS.proximityAvailable() || (APDS.readProximity() == 0)) {}
  }
}
```



## Materiales 
-Arduino Nano 33 BLE 

-Computador 

-Colores para registrar 

-Internet 


 ### Software : 
Arduino ID


## Referentes y recursos adicionales
Código para activar LED 
https://docs.arduino.cc/tutorials/nano-33-ble-sense/rgb-sensor


Carpetas a instalar en el programa Arduino:

-Arduino_APDS 9960

-Arduino_tensorflowlite

-Harvard_tinyMLX


Link para entrenar algoritmo de Google Colab sin inetrvenir:
https://colab.research.google.com/github/arduino/ArduinoTensorFlowLiteTutorials/blob/master/FruitToEmoji/FruitToEmoji.ipynb


Link de Google Colab modificado y resultante:
https://colab.research.google.com/drive/1sNo9rCZT6jovKp3YDQ1F2C3YG9fRIbwN


## Conclusiones y aprendizajes
-Al realizar el entrenamiento, utilizamos colores RGB por lo que cuando identificamos colores en el entono real, presenta algunas dificultades para reconocer ciertos espectros. El algoritmo es muy fan de los elemento FUEGO y el VIENTO...Si eres de TIERRA wow y si eres AGUA doble wow. 

-Es importante una buena luminosidad en el entorno para que capte mejor los colores.

- Fue sumamente importante la colaboración entre compañeres del curso de intelgencia artificial. Tanto entre nosotras como grupo y la ayuda y guía recibida desde Aaron, J y el grupo de Amelia.
  
+Aprendizaje: Registro de error en línea de datos de archivo csv de la captura cromática correspondiente a fuego.
Al momento de ejecutar las celdas en colab solo logró resultados en 2 colores ya que el tercero presentaba un error en una línea de código y en consecuencia desde la carpeta con el registro cromático defectuoso en adelante se detuvo.

![9CA9C69B-D850-4052-A39B-D32E635DE40E](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/6d85d823-9e42-460f-bc12-8eb665a4e141)

Intentamos ocupar la cámara más grande pensando en tener resultados mas precisos al momento de identificar, pero no fue posible ya que el código que estamos utilizando esta hecho para la cámara integrada en el arduino.

![IMG_8051](https://github.com/sofantasyy/audiv027-2023-2/assets/142052341/2fc41cf0-0345-4aab-9c5c-6d3113c0b363)
