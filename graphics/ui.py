import tkinter as tk

from .lib import BOARD_ZOOM, BACKGROUND_COLOR, FONT, SIZE


class UI(object):
    def __init__(self, root):
        self.root = root

        self.init_panels()
        self.init_buttons()

    # разделение экрана на 3 составные панели
    def init_panels(self):
        width = SIZE
        height = (SIZE - BOARD_ZOOM[1]) // 2

        self.panel_top = tk.Frame(self.root, width=width, height=height)
        self.panel_top.configure(background=BACKGROUND_COLOR)
        self.panel_top.place(relx=0, rely=0, anchor='nw')

        self.panel_base = tk.Frame(self.root, width=width, height=BOARD_ZOOM[1])
        self.panel_base.configure(background=BACKGROUND_COLOR)
        self.panel_base.place(relx=0.5, rely=0.5, anchor='center')

    # обновление инфы по игре
    def update_info(self, players, timers, mess):
        score1 = f"Time: {timers[0]}s\nP1: {players[0].captures} army captured."
        score2 = f"Time: {timers[1]}s\nP2: {players[1].captures} army captured."

        font = {'width': 25, 'height': 2, 'font': FONT, 'background': BACKGROUND_COLOR}

        label1 = tk.Label(self.panel_top, text=score1, **font)
        label1.place(relx=0.1, rely=0.65, anchor='nw')
        label2 = tk.Label(self.panel_top, text=score2, **font)
        label2.place(relx=0.9, rely=0.65, anchor='ne')
        title = tk.Label(self.panel_top, text=mess, **font)
        title.place(relx=0.5, rely=0.85, anchor='center')

    # инициализация кнопок
    def init_buttons(self):
        btn_config = {'width': 8, 'height': 2, 'font': FONT}

        r_btn = tk.Button(self.panel_top, text="Restart")
        r_btn.configure(command=self.root.restartCallback, **btn_config)
        r_btn.place(relx=0.35, rely=0.5, anchor='s')

        u_btn = tk.Button(self.panel_top, text="Undo")
        u_btn.configure(command=self.root.undoCallback, **btn_config)
        u_btn.place(relx=0.45, rely=0.5, anchor='s')

        h_btn = tk.Button(self.panel_top, text="Prompt")
        h_btn.configure(command=self.root.helpCallback, **btn_config)
        h_btn.place(relx=0.55, rely=0.5, anchor='s')

        q_btn = tk.Button(self.panel_top, text="Quit")
        q_btn.configure(command=self.root.quitCallback, **btn_config)
        q_btn.place(relx=0.65, rely=0.5, anchor='s')
