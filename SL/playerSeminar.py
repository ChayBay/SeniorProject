
class playerObj():
    def __init__(self, playerX, playerY):
        self.x = playerX
        self.y = playerX
        
        self.prevX = playerX
        self.prevY = playerY

        self.prog = 0

    def __str__(self):
        return f"players coords ({self.x}, {self.y})"
        
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

