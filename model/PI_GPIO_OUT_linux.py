import gpiod
import time
from gpiod.line import Direction, Value

with gpiod.request_lines(
    "/dev/gpiochip0",
    consumer="gpio_output",
    config={17: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE)}
) as request:
    request.set_value(17, Value.ACTIVE)
    time.sleep(1)
    request.set_value(17, Value.INACTIVE)
