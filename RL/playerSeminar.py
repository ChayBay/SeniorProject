
class playerObj():
    def __init__(self, playerX, playerY):
        self.x = playerX
        self.y = playerY
        
        self.prevX = playerX
        self.prevY = playerY

        self.prog = 0

    def __str__(self):
        return f"obj coords ({self.x}, {self.y})"
        
