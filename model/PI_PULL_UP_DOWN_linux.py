import gpiod
from gpiod.line import Direction, Bias
import time 

with gpiod.request_lines(
    "/dev/gpiochip0",
    consumer="gpio_pull",
    config={17: gpiod.LineSettings(
        direction=Direction.INPUT,
        bias=Bias.PULL_UP
    )}
) as request:
    while True:
        value = request.get_value(17)
        print(f'GPIO 17 value: {value}')
        time.sleep(1)