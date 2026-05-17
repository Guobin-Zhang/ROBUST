// rpi5_gpio_driver.c
// Compile: gcc -O3 -shared -fPIC -o rpi5_gpio.so rpi5_gpio_driver.c
// Requires root access for GPIO mmap

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>

#define GPIO_BASE       0xFE200000  // RPi5 BCM2712 GPIO base
#define BLOCK_SIZE      4096

#define SCLK_PIN        11
#define MOSI_PIN        10
#define MISO_PIN        9
#define CS_PIN          8
#define IRQ_PIN         25

// GPIO Register offsets
#define GPFSEL0         0x00
#define GPSET0          0x1C
#define GPCLR0          0x28
#define GPLEV0          0x34

volatile uint32_t *gpio_base = NULL;

static inline void gpio_set_output(int pin) {
    int reg = pin / 10;
    int shift = (pin % 10) * 3;
    gpio_base[GPFSEL0 + reg] &= ~(7 << shift);
    gpio_base[GPFSEL0 + reg] |= (1 << shift);
}

static inline void gpio_set_input(int pin) {
    int reg = pin / 10;
    int shift = (pin % 10) * 3;
    gpio_base[GPFSEL0 + reg] &= ~(7 << shift);
}

static inline void gpio_write(int pin, int val) {
    if (val)
        gpio_base[GPSET0] = (1 << pin);
    else
        gpio_base[GPCLR0] = (1 << pin);
}

static inline int gpio_read(int pin) {
    return (gpio_base[GPLEV0] >> pin) & 1;
}

// Nanosecond delay using busy-wait (no syscall overhead)
static inline void ns_delay(int ns) {
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    long target_ns = start.tv_nsec + ns;
    do {
        clock_gettime(CLOCK_MONOTONIC, &now);
    } while (now.tv_nsec < target_ns && now.tv_sec == start.tv_sec);
}

// Optimized bit-bang SPI transfer
uint8_t spi_transfer_byte(uint8_t tx_data) {
    uint8_t rx_data = 0;
    
    for (int i = 7; i >= 0; i--) {
        // Setup MOSI
        gpio_write(MOSI_PIN, (tx_data >> i) & 1);
        
        // SCLK rising edge
        gpio_write(SCLK_PIN, 1);
        
        // Sample MISO at middle of clock
        ns_delay(50);  // 100ns period = 10MHz
        
        rx_data = (rx_data << 1) | gpio_read(MISO_PIN);
        
        // SCLK falling edge
        gpio_write(SCLK_PIN, 0);
        
        ns_delay(50);
    }
    
    return rx_data;
}

// Frame transfer with CRC
int spi_transfer_frame(uint8_t cmd, uint8_t chip, uint16_t row, uint16_t col, 
                       uint32_t data, uint8_t *resp, int resp_len) {
    uint8_t frame[11];
    uint8_t crc = 0;
    
    // Build frame
    frame[0] = 0xAA;  // Sync high
    frame[1] = 0x55;  // Sync low
    frame[2] = cmd;
    frame[3] = chip;
    frame[4] = (row >> 8) & 0xFF;
    frame[5] = row & 0xFF;
    frame[6] = (col >> 8) & 0xFF;
    frame[7] = col & 0xFF;
    frame[8] = (data >> 24) & 0xFF;
    frame[9] = (data >> 16) & 0xFF;
    frame[10] = (data >> 8) & 0xFF;
    // frame[11] = data & 0xFF;  // Extended to 4 bytes if needed
    
    // Calculate CRC (XOR)
    for (int i = 0; i < 11; i++) {
        crc ^= frame[i];
    }
    
    // Assert CS
    gpio_write(CS_PIN, 0);
    ns_delay(100);  // Setup time
    
    // Transfer frame
    for (int i = 0; i < 11; i++) {
        spi_transfer_byte(frame[i]);
    }
    spi_transfer_byte(crc);
    
    // Read response
    for (int i = 0; i < resp_len; i++) {
        resp[i] = spi_transfer_byte(0x00);  // Dummy bytes for clock
    }
    
    // Deassert CS
    gpio_write(CS_PIN, 1);
    ns_delay(100);  // Hold time
    
    return 0;
}

// Initialization
int rpi5_gpio_init(void) {
    int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (mem_fd < 0) {
        perror("Failed to open /dev/mem");
        return -1;
    }
    
    gpio_base = mmap(NULL, BLOCK_SIZE, PROT_READ | PROT_WRITE, 
                     MAP_SHARED, mem_fd, GPIO_BASE);
    close(mem_fd);
    
    if (gpio_base == MAP_FAILED) {
        perror("mmap failed");
        return -1;
    }
    
    // Configure pins
    gpio_set_output(SCLK_PIN);
    gpio_set_output(MOSI_PIN);
    gpio_set_input(MISO_PIN);
    gpio_set_output(CS_PIN);
    gpio_set_input(IRQ_PIN);
    
    // Initial states
    gpio_write(SCLK_PIN, 0);
    gpio_write(MOSI_PIN, 0);
    gpio_write(CS_PIN, 1);  // CS inactive high
    
    return 0;
}

// Cleanup
void rpi5_gpio_cleanup(void) {
    if (gpio_base) {
        munmap((void*)gpio_base, BLOCK_SIZE);
    }
}