import tkinter as tk

from PIL import Image, ImageEnhance, ImageTk

from .lib import (BOARD_OFFSET, BOARD_ZOOM, BACKGROUND_COLOR, ARMY_SIZE, SIZE, transform_coords)


class Canvas(object):
    def __init__(self, root):
        self.root = root
        self.load_tkImages()
        self.load_canvas()

        self.hover_stone = None
        self.color = None

        self.canvas.bind("<ButtonPress-1>", self.mouse_lkm)
        self.canvas.bind("<Motion>", self.mouse_move)

    # выгрузка изображений
    def load_tkImages(self):
        error = Image.open("./img/error.png").resize(ARMY_SIZE, Image.ANTIALIAS)
        self.tkError = ImageTk.PhotoImage(error)
        board = Image.open("./img/board.png").resize(BOARD_ZOOM, Image.ANTIALIAS)
        self.tkBoard = ImageTk.PhotoImage(board)
        horde = Image.open("./img/white.png").resize(ARMY_SIZE, Image.ANTIALIAS)
        self.tkHorde = ImageTk.PhotoImage(horde)
        horde.putalpha(ImageEnhance.Brightness(horde.split()[3]).enhance(0.5))
        self.tkHordeAlpha = ImageTk.PhotoImage(horde)
        alliance = Image.open("./img/black.png").resize(ARMY_SIZE, Image.ANTIALIAS)
        self.tkAlliance = ImageTk.PhotoImage(alliance)
        alliance.putalpha(ImageEnhance.Brightness(alliance.split()[3]).enhance(0.5))
        self.tkAllianceAlpha = ImageTk.PhotoImage(alliance)

    # выгрузка канваса
    def load_canvas(self):
        width = SIZE
        height = BOARD_ZOOM[1]

        self.canvas = tk.Canvas(self.root, width=width, height=height)
        self.canvas.configure(background=BACKGROUND_COLOR)
        self.canvas.place(relx=0.5, rely=0.499, anchor='center')

    # обновление доски по запросу
    def update_board(self, Field):
        self.color = Field.cnt_player
        # self.canvas.delete("all")
        self.canvas.create_image(*BOARD_OFFSET, anchor='nw', image=self.tkBoard)
        board = Field.board
        size = Field.size
        for i in range(size):
            for j in range(size):
                stone = board[i][j]
                if stone != 0:
                    stone = self.tkAlliance if stone == 2 else self.tkHorde
                    offset = transform_coords([j, i])
                    self.canvas.create_image(*offset, anchor="nw", image=stone)

        if len(Field.move_history) > 0:
            last_move = Field.move_history[-1]
            last_move = last_move[::-1]
            offset = transform_coords(last_move) + ARMY_SIZE // 2
            radius = ARMY_SIZE[0] // 2
            self.canvas.create_oval(offset[0] - radius,
                                    offset[1] - radius,
                                    offset[0] + radius,
                                    offset[1] + radius,
                                    outline="white", width=2)


    # отображение незаконных ходов
    def show_illegal_move(self, coords):
        offset = transform_coords(coords)
        self.canvas.create_image(*offset, anchor="nw", image=self.tkError)

    # ресет перед новой игрой
    def reset(self):
        self.hover_stone = None

    # эвент движения мыши
    def mouse_move(self, event):
        if not self.readyForInput():
            return
        coords = ((event.x, event.y) - BOARD_OFFSET) // (BOARD_ZOOM * 0.052)
        if (coords > 18).any() or (coords < 0).any():
            return
        self.canvas.delete(self.hover_stone)
        stone = self.tkAllianceAlpha if self.color == 1 else self.tkHordeAlpha

        config = {'anchor': "nw", 'image': stone}
        offset = transform_coords(coords)
        self.hover_stone = self.canvas.create_image(*offset, **config)

    # эвент лкм
    # чекаем что не улетаем за пределы карты и уравниваем значения
    def mouse_lkm(self, event):
        if not self.readyForInput():
            return
        coords = ((event.x, event.y) - BOARD_OFFSET) // (BOARD_ZOOM * 0.052)
        if (coords > 18).any() or (coords < 0).any():
            return

        self.mouseDownCallback([int(x) for x in coords])
