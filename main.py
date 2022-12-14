import argparse
import json
import os.path
import sys
from typing import Tuple

from globals import *
from messages import Messages
from board.board import Field
# from rules import Rules
# from game_assist import GameAssist

from graphics import Movie
from utils import EmptyException
from ai_engine import ai_engine


def parse_args() -> argparse.Namespace:
	"""
	Parsing arguments and options

	:return: parsed arguments and options
	"""
	parser = argparse.ArgumentParser()
	# parser.add_argument('data',
	# 					type=str,
	# 					help="Path to player's moves notation file",
	# 					default='data/data.csv')
	parser.add_argument('--statistics', '-s',
						dest='stat',
						action="store_true",
						default=False,
						help='Save moves notation to file data/data.csv ('
							 'default=False)')
	parser.add_argument('--terminal', '-t',
						dest='term',
						action="store_true",
						default=False,
						help='use terminal mode (default=False)')
	parser.add_argument('--player1', '-p1',
						type=str,
						dest='p1',
						default='human',
						help="Player1's behaviour should be in: "
							 'human(default), minimax, alpha_beta, ...')
	parser.add_argument('--player2', '-p2',
						type=str,
						dest='p2',
						default='AI',
						help="Player2's behaviour should be in: "
							 'human(default), minimax, alpha_beta, ...')
	parser.add_argument('--log', '-l',
						dest='log',
						action="store_true",
						default=False,
						help='Show moves log at the end of the game ('
							 'default=False)')

	# parser.add_argument('--depth', '-d',
	# 					dest='depth',
	# 					type=int,
	# 					default=10,
	# 					help='Search Tree depth for MiniMax (default=10)')
	# parser.add_argument('--heuristic', '-h',
	# 					dest='heur',
	# 					action="store_true",
	# 					default=False,
	# 					help="Use heuristic function (default=False)")
	return parser.parse_args()


	################### НАПИШУ ПОТОМ №№№№№№№№№№№№№№№№№№№№№№№№
	# def validate_args(args: argparse.Namespace) -> None:
	# """
	# Validate command-line arguments of the program
	#
	# :param args: command-line arguments of the program
	# :return: Exit if smth wrong with parameters, pass otherwise
	# """

def parse_config(tag: str) -> dict:
	try:
		with open('./players_config.json', 'r') as f:
			conf = json.load(f)
			return conf.get(tag, {})
	except FileNotFoundError:
		return {}

def main(debug=False):
	"""
	Main gomoku program
	"""
	exceptions_to_catch = EmptyException if debug else Exception
	try:
		args = parse_args()
		p1_conf, p2_conf = parse_config("player1"), parse_config("player2")
		# validate_args(args)
		# С разными вариантами Player-ов нужно будет переделать,
		# пока заглушка
		game = Field(filename=None, players=[])
		players = [get_player(args.p1, [1, 2, game], p1_conf), get_player(args.p2, [2, 1, game], p2_conf)]

		game.ai_bot = get_player('AI', [0, 0, game], {}) # TODO удалить

		if args.p1 == "human":
			players[0].human = 1
		if args.p2 == "human":
			players[1].human = 1
		game.players = players

		if args.term or debug:
			game.start_terminal(args.log)
		else:
			Movie(game, args.log, parse_config("bot"))

		ai_engine.terminate_processes()

	except exceptions_to_catch as e:
		Messages(e.__doc__).error_()
		ai_engine.terminate_processes()
		sys.exit(0)


if __name__ == "__main__":
	try:
		main(False)
	except KeyboardInterrupt:
		ai_engine.terminate_processes()

