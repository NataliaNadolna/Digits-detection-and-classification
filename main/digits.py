class Digit():
    def __init__(self, img, path: str):
        self.img = img
        self.path = path

class Number(list):
    def __init__(self):
        super().__init__()