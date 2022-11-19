import argparse
import sys

from globals import *
from messages import Messages
from board import Field
# from rules import Rules
# from game_assist import GameAssist

from graphics import Movie


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


def main():
	"""
	Main gomoku program
	"""
	try:
		args = parse_args()
		# validate_args(args)
		# С разными вариантами Player-ов нужно будет переделать,
		# пока заглушка
		game = Field(filename=None, players=[])
		players = [COMPETITORS[args.p1](1, 2, game), COMPETITORS[args.p2](2, 1, game)]
		# players = [args.p1, args.p2]
		# game = GameAssist(board=Field(filename=None), players=players)
		game.players = players

		if args.term:
			game.start_terminal()
		else:
			Movie(game)

	except Exception as e:
		Messages(e.__doc__).error_()
		sys.exit(0)


if __name__ == "__main__":
	main()
