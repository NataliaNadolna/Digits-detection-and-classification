import random

def generate_color(colors: dict):
    random_color = []
    for color in colors:
        min = colors[color][0]
        max = colors[color][1]
        c = random.randint(min, max)
        random_color.append(c)
    return random_color