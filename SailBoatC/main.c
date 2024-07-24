#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/gpio.h"
#include "hardware/pwm.h"
#include "pico/time.h"


#define Pitch 16
#define Yaw 17
#define PWM_PIN 13
#define Trigger 15

int Pitch_Angle = 750;
int Yaw_Angle = 750;
double distancia;
int cambio =50;
int value=0;

char userInput[10];
//ff

void initPWM(uint8_t gpio, uint16_t frec) {
    uint slice = pwm_gpio_to_slice_num(gpio);
    pwm_config cfg = pwm_get_default_config();
    pwm_config_set_clkdiv(&cfg, (float)SYS_CLK_KHZ / 500);
    pwm_config_set_clkdiv_mode(&cfg, PWM_DIV_FREE_RUNNING);
    pwm_config_set_wrap(&cfg, (500000) / (frec));
    pwm_init(slice, &cfg, true);
}

void initRot(uint8_t PWM_GPIO) {
    gpio_set_function(PWM_GPIO, GPIO_FUNC_PWM);
    pwm_set_chan_level(pwm_gpio_to_slice_num(PWM_GPIO), pwm_gpio_to_channel(PWM_GPIO), 820);
    sleep_ms(3000);
    pwm_set_chan_level(pwm_gpio_to_slice_num(PWM_GPIO), pwm_gpio_to_channel(PWM_GPIO), 750);
}

double medirDistancia() {
    int counts = 0;
    double Distancia_sample = 0;
    double timeFinal;
    double time1, time2;

    while (counts < 6) {
        if (counts > 0) {
            while (gpio_get(PWM_PIN) == 0) {}
            time1 = to_us_since_boot(get_absolute_time());

            while (gpio_get(PWM_PIN) != 0) {}
            time2 = to_us_since_boot(get_absolute_time());
            timeFinal = (time2 - time1) / 10;
            Distancia_sample += timeFinal;
            counts++;
        } else {
            while (gpio_get(PWM_PIN) == 0) {}

            while (gpio_get(PWM_PIN) != 0) {}
            counts++;
        }
    }

    return Distancia_sample / 6 - 20.0;
}

int main() {
    stdio_init_all();
    initPWM(Pitch, 50);
    gpio_init(PWM_PIN);
    gpio_set_dir(PWM_PIN, GPIO_IN);
    gpio_init(Trigger);
    gpio_set_dir(Trigger, GPIO_OUT);
    gpio_put(Trigger, false);

    if (pwm_gpio_to_slice_num(Pitch) != pwm_gpio_to_slice_num(Yaw)) {
        initPWM(Yaw, 50);
    }

    initRot(Pitch);
    initRot(Yaw);
    char USBInput;

    while (true) {
        USBInput = getchar();
        switch (USBInput) {
            case 'a':
                printf("p\n");
                scanf("%s", userInput); // Leer la entrada del usuario como una cadena de caracteres
                
                if (sscanf(userInput, "%d", &value) == 1) { // Convertir la entrada del usuario a un entero
                    printf("%d\n", value);
                } 
                Pitch_Angle += value;
                if (Pitch_Angle >= 900) {
                    Pitch_Angle = 900;
                }
                break;
            case 'u':
                printf("p\n");
                scanf("%s", userInput); // Leer la entrada del usuario como una cadena de caracteres
              
                if (sscanf(userInput, "%d", &value) == 1) { // Convertir la entrada del usuario a un entero
                    printf("%d\n", value);
                } 
                Pitch_Angle -= value;
                if (Pitch_Angle <= 300) {
                    Pitch_Angle = 300;
                }
                break;
            case 'i':
                printf("y\n");
                scanf("%s", userInput); // Leer la entrada del usuario como una cadena de caracteres
                
                if (sscanf(userInput, "%d", &value) == 1) { // Convertir la entrada del usuario a un entero
                    printf("%d\n", value);
                } 
                Yaw_Angle += value;
                if (Yaw_Angle >= 900) {
                    Yaw_Angle = 900;
                }
                break;
            case 'd':
                printf("y\n");
                scanf("%s", userInput); // Leer la entrada del usuario como una cadena de caracteres
                
                if (sscanf(userInput, "%d", &value) == 1) { // Convertir la entrada del usuario a un entero
                    printf("%d\n", value);
                } 
                Yaw_Angle -= value;
                
                if (Yaw_Angle <= 500) {
                    Yaw_Angle = 500;
                }
                break;

            case 'r':
                Yaw_Angle=750;
                Pitch_Angle=750;
                break;

            case 'l':
                distancia = medirDistancia();
                printf("%.2f\n", distancia);
                //Yaw_Angle=750;
                //Pitch_Angle=750;
                break;
            default:
                break;
        }

        pwm_set_gpio_level(Pitch, Pitch_Angle);
        pwm_set_gpio_level(Yaw, Yaw_Angle);
        
    }
}
