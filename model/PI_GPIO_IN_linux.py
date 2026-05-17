import gpiod
from gpiod.line import Direction

with gpiod.request_lines(
    "/dev/gpiochip0",
    consumer="gpio_input",
    config={17: gpiod.LineSettings(direction=Direction.INPUT)}
) as request:
    while True:
        value = request.get_value(17)
        print(f'GPIO 17 value: {value}')
