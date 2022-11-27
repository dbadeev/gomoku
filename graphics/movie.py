import time
import tkinter as tk

# from agent import Agent
# from game_assist import GameAssist
from board.board import Field
from .canvas import Canvas
from .ui import UI
from .lib import BACKGROUND_COLOR, SIZE
from player.human_player import HumanPlayer
from player.ai_player import GomokuAIPlayer


# класс отрисовки
class Movie(object):
	def __init__(self, Field):
		# ссылка на все основные элементы игрового поля
		# self.GameAssist = GameAssist
		self.Field = Field

		# подключение и настройка ткинтера
		self.root = tk.Tk()
		self.root.title("GOMOKU_21 by nmadelei & loram")
		self.root.geometry(str(SIZE) + 'x' + str(SIZE))
		self.root.resizable(False, False)
		self.root.configure(background=BACKGROUND_COLOR)

		self.timers = [0, 0]
		# подписки и инициализация ui скрипта
		self.root.restartCallback = self.on_click_restart
		self.root.helpCallback = self.on_click_help
		self.root.quitCallback = self.on_click_quit
		self.root.undoCallback = self.on_click_undo
		self.ui = UI(self.root)

		# подписки пд и инициализация канваса
		self.canvas = Canvas(self.ui.panel_base)
		self.canvas.readyForInput = self.readyForInput
		self.canvas.mouseDownCallback = self.mouse_lkm

		# инициализация переменных
		self.on_click_restart()

		# запускаем бесконечный цикл
		self.update_logic()
		self.update_labels()
		self.root.mainloop()

	# отображение игровой инфы каждую секунду
	def update_labels(self):
		message = f"Turn {self.Field.turn}"
		if self.Field.winner is not None:
			message = f"P{self.Field.winner.color} won."
		self.ui.update_info(self.Field.players, self.timers, message)
		self.root.after(1000, self.update_labels)

	# каждый тик
	def update_logic(self):
		# если предыдущая итерация закончилась
		if not self.b_update:
			self.canvas.update_board(self.Field)
			self.b_update = True
			self.root.after(1, self.update_logic)
			return

		# если игра окончена уходим в бесконечность
		if self.over:
			self.root.after(1, self.update_logic)
			return

		# выбор AI
		# self.Field.begin = time.time()
		cnt_player = self.Field.cnt_player
		player = self.Field.players[cnt_player]
		# is_notation = self.Field.notation and self.Field.notation.running()

		# ход AI
		# if is_notation:
		# 	move = self.Field.notation.get_move()
		# elif isinstance(player, Agent):
		# 	self.playerInput = False
		# 	move = player.find_move(self.Field)
		# else:
		# 	self.playerInput = True
		# 	if self.move is None:
		# 		self.root.after(1, self.update_logic)
		# 		return
		# 	move = self.move[::-1]
		if isinstance(player, GomokuAIPlayer):
			self.Field.begin = time.time()
			move = player.get_move(self.Field)
			self.timers[cnt_player] = round(time.time() - self.Field.begin, 2)
			self.Field.begin = time.time()
		elif isinstance(player, HumanPlayer):
			# self.Field.begin = time.time()
			self.playerInput = True
			if self.move is None:
				self.root.after(1, self.update_logic)
				return
			# self.Field.begin = time.time()
			move = self.move[::-1]
			if self.Field.begin < 0:
				self.Field.begin = time.time()
			self.timers[cnt_player] = round(time.time() - self.Field.begin, 2)
			self.Field.begin = time.time()

		# если скорость работы бота медленнее лимита то противоположный игрок выйграл
		# if self.timers[cnt_player] > self.Field.time_limit:
		# 	self.Field.winner = self.Field.players[1 - cnt_player]

		# если точка на доске уже занята
		if not self.Field.is_empty(*move):
			self.root.after(1, self.update_logic)
			return

		# ход и проверка на незаконный ход
		if not self.Field.place_on_board(move):
			if move not in self.illegal_moves:
				self.illegal_moves.append(move)
				self.canvas.show_illegal_move(move[::-1])
			self.root.after(1, self.update_logic)
			return

		# если последнее условие прошло ровно то подготовка рекурсии к следующему ходу
		# self.playerInput = not self.playerInput
		self.move = None
		self.illegal_moves = []
		self.b_update = False
		self.over = self.Field.winner != None

		# self.root.after(250 if is_notation else 1, self.update_logic)
		self.root.after(1, self.update_logic)
		return

	def readyForInput(self):
		return self.playerInput and not self.over

	def mouse_lkm(self, move):
		self.move = move

	def on_click_restart(self):
		# self.Board.reset_all()
		self.Field.reset_all()

		self.canvas.reset()

		# ??
		self.playerInput = False
		# следующий ход?
		self.b_update = False
		# конец игры?
		self.over = False
		# текущий ход
		self.move = None
		# незаконные ходы
		self.illegal_moves = []
		# тайминги
		self.timers = [0, 0]

		self.canvas.update_board(self.Field)
		players = self.Field.players
		message = f"Turn {self.Field.turn}"
		self.ui.update_info(players, self.timers, message)

	# выход из приложения
	def on_click_quit(self):
		self.root.destroy()

	# помощь в виде хода компа
	def on_click_help(self, iteration=0):
		if not self.readyForInput():
			return
		self.move = self.Field.move_hint()
		self.playerInput = False

	# ctrl + z откат последнего хода
	def on_click_undo(self):
		if not self.readyForInput():
			return
		for _ in range(2):
			if len(self.Field.move_history) > 0:
				self.Field.undo_move()
				self.Field.turn -= 1
		for player in self.Field.players:
			if hasattr(player, 'undo_scores'):
				player.color_scores = player.undo_scores
		self.playerInput = False
		self.canvas.update_board(self.Field)
