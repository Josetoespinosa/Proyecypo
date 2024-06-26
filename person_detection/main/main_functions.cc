#include "main_functions.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "ball_classification_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"
#include "esp_psram.h"
#include "esp_system.h" 
#include "esp_task_wdt.h"
#include "led_strip.h"
#include "driver/gpio.h"
#include "driver/ledc.h"


//CONFIG_ESP_TASK_WDT_TIMEOUT_S(10);

#define BLINK_GPIO gpio_num_t(4)

#define SERVO_PIN GPIO_NUM_13

// Define la frecuencia del PWM
#define SERVO_PWM_FREQ 50 // Frecuencia típica para servos (50Hz)

// Define los ángulos de movimiento del servo
#define SERVO_MIN_ANGLE 0
#define SERVO_MAX_ANGLE 180

// Define los valores del pulso en microsegundos para los ángulos
#define SERVO_MIN_PULSEWIDTH 500  // Pulso de 500us para el ángulo mínimo
#define SERVO_MAX_PULSEWIDTH 2400// 20000 ticks, 20ms

uint32_t angle_to_duty(int angle) {
    int pulsewidth = SERVO_MIN_PULSEWIDTH + ((SERVO_MAX_PULSEWIDTH - SERVO_MIN_PULSEWIDTH) * angle) / (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE);
    uint32_t duty = (pulsewidth * 8192) / (1000000 / SERVO_PWM_FREQ);
    return duty;
}

void set_servo_angle(int angle) {
    uint32_t duty = angle_to_duty(angle);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
    printf("Moviendo servo al ángulo %d\n", angle);
}

int N = 2097152;

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 40 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif

static int kTensorArenaSize = 400000 + scratchBufSize;  // Reduced size for testing
static uint8_t *tensor_arena;
}  // namespace

void setup() {
  // Initialize PSRAM
  if (esp_psram_get_size() == 0) {
    printf("PSRAM not found\n");
    return;
  }

  ledc_timer_config_t ledc_timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_13_BIT, // Resolución de 13 bits
        .timer_num = LEDC_TIMER_0,
        .freq_hz = SERVO_PWM_FREQ,    // Frecuencia del PWM
        .clk_cfg = LEDC_AUTO_CLK,
        .deconfigure = 0
    };
    ledc_timer_config(&ledc_timer);

    // Configura el canal PWM
    ledc_channel_config_t ledc_channel = {
        .gpio_num = SERVO_PIN,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = LEDC_CHANNEL_0,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = LEDC_TIMER_0,
        .duty = 0,
        .hpoint = 0,
        .flags = {
            .output_invert = 0,
        }
    };
    ledc_channel_config(&ledc_channel);


    // Mueve el servo a un ángulo específico (por ejemplo, 90 grados)
    set_servo_angle(90);


  // Initialize LED strip
  gpio_reset_pin(BLINK_GPIO);
  gpio_set_direction(BLINK_GPIO, GPIO_MODE_OUTPUT);//

  // Print memory information

  printf("Total heap size: %d\n", heap_caps_get_total_size(MALLOC_CAP_8BIT));
  printf("Free heap size: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT));
  printf("Total PSRAM size: %d\n", esp_psram_get_size());
  printf("Free PSRAM size: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // Initialize model
  model = tflite::GetModel(model_quant_ball_clasification6_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Allocate tensor arena in PSRAM
  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  printf("Free heap size after allocation: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT));
  printf("Free PSRAM size after allocation: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // Define MicroMutableOpResolver and add required operations
  static tflite::MicroMutableOpResolver<7> micro_op_resolver;
  micro_op_resolver.AddQuantize();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMaxPool2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddDequantize();

  // Build and allocate the interpreter
  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE
// The name of this function is important for Arduino compatibility.
void loop() {
  //int64_t start_time_preparation = esp_timer_get_time();
  // Get image from provider.
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }

  //int64_t end_time_preparation = esp_timer_get_time();
  //int64_t preparation_time = end_time_preparation - start_time_preparation;
  //MicroPrintf( "Preparation time: %lld microseconds", preparation_time);
  //int64_t start_time_inference = esp_timer_get_time();
  // Run the model on this input and make sure it succeeds.

  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }


  //int64_t end_time_inference = esp_timer_get_time();
  //int64_t inference_time = end_time_inference - start_time_inference;
  //MicroPrintf( "Inference time: %lld microseconds", inference_time);


  //int64_t start_time_processing = esp_timer_get_time();
  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t basketball_score = output->data.uint8[kBaketballIndex];
  int8_t soccer_score = output->data.uint8[kSoccerIndex];
  int8_t rugby_score = output->data.uint8[kRugbyIndex];
  int8_t table_tennis_score = output->data.uint8[kTableTennisIndex];
  int8_t tennis_score = output->data.uint8[kTennisIndex];
  int8_t volleyball_score = output->data.uint8[kVolleyballIndex];

  //int64_t end_time_processing = esp_timer_get_time();
  //int64_t processing_time = end_time_processing - start_time_processing;
  //MicroPrintf("Response processing time: %lld microseconds", processing_time);

  //int64_t start_time_response = esp_timer_get_time();

  float basketball_score_f =
      (basketball_score - output->params.zero_point) * output->params.scale;
  
  float soccer_score_f =
      (soccer_score - output->params.zero_point) * output->params.scale;

  float rugby_score_f =
      (rugby_score - output->params.zero_point) * output->params.scale;
  
  float table_tennis_score_f =
      (table_tennis_score - output->params.zero_point) * output->params.scale;
  
  float tennis_score_f =
      (tennis_score - output->params.zero_point) * output->params.scale;

  float volleyball_score_f =
      (volleyball_score - output->params.zero_point) * output->params.scale;

  // Respond to detection
  int times = RespondToDetection(basketball_score_f, soccer_score_f, rugby_score_f, table_tennis_score_f, tennis_score_f, volleyball_score_f);

  // Blink LED
  for (int i = 0; i < times; i++) {
    gpio_set_level(BLINK_GPIO, 1);
    vTaskDelay(100 / portTICK_PERIOD_MS);
    gpio_set_level(BLINK_GPIO, 0);
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }

  int angle = 60*times-30;
  set_servo_angle(angle);
  //servo1.write(angle);

  
  // timeit("RespondToDetection", [&]() {
  //   RespondToDetection(basketball_score_f, soccer_score_f, rugby_score_f, table_tennis_score_f, tennis_score_f, volleyball_score_f);
  // });
  //int64_t end_time_response = esp_timer_get_time();
  //int64_t response_time = end_time_response - start_time_response;
  //MicroPrintf("Response time: %lld microseconds", response_time);

  //int64_t end_processing = esp_timer_get_time();
  //int64_t total_time = end_processing - start_time_preparation;
  //MicroPrintf("Total time: %lld microseconds", total_time);
  vTaskDelay(800);
}
#endif

#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long softmax_total_time;
  extern long long dc_total_time;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
  extern long long add_total_time;
  extern long long mul_total_time;
#endif

void run_inference(void *ptr) {
  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    input->data.int8[i] = ((uint8_t *) ptr)[i] ^ 0x80;
    printf("%d, ", input->data.int8[i]);
  }

#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time = %lld\n", total_time / 1000);
  //printf("Softmax time = %lld\n", softmax_total_time / 1000);
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("DC time = %lld\n", dc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  printf("add time = %lld\n", add_total_time / 1000);
  printf("mul time = %lld\n", mul_total_time / 1000);

  /* Reset times */
  total_time = 0;
  //softmax_total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
#endif

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t basketball_score = output->data.uint8[kBaketballIndex];
  int8_t soccer_score = output->data.uint8[kSoccerIndex];
  int8_t rugby_score = output->data.uint8[kRugbyIndex];
  int8_t table_tennis_score = output->data.uint8[kTableTennisIndex];
  int8_t tennis_score = output->data.uint8[kTennisIndex];
  int8_t volleyball_score = output->data.uint8[kVolleyballIndex];

  float basketball_score_f =
      (basketball_score - output->params.zero_point) * output->params.scale;
  
  float soccer_score_f =
      (soccer_score - output->params.zero_point) * output->params.scale;

  float rugby_score_f =
      (rugby_score - output->params.zero_point) * output->params.scale;
  
  float table_tennis_score_f =
      (table_tennis_score - output->params.zero_point) * output->params.scale;
  
  float tennis_score_f =
      (tennis_score - output->params.zero_point) * output->params.scale;

  float volleyball_score_f =
      (volleyball_score - output->params.zero_point) * output->params.scale;

  // Respond to detection
  RespondToDetection(basketball_score_f, soccer_score_f, rugby_score_f, table_tennis_score_f, tennis_score_f, volleyball_score_f);
}
