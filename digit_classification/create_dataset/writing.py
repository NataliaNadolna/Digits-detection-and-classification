from functions import generate_color
import random

class Writing_style():
    def __init__(self, colors: list, position: dict, fonts: list, size: list, thickness: list, index: int):
        self.color = generate_color(colors)

        up_down = random.randint(position["down"], position["up"])
        left_right = random.randint(position["left"], position["right"])
        self.coords = (left_right, up_down)

        self.font = fonts[index % len(fonts)]
        self.thickness = thickness[index % len(thickness)]
        self.size = random.uniform(size[0], size[1])