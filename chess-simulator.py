from queue import Queue
import chess, random, _thread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

class Simulation:
	def __init__():
		# initialize data storage for number of visits per square
		self.white_visits = np.zeros((64, 6), dtype=int)
		self.black_visits = np.zeros((64, 6), dtype=int)
		self.piece_map = {1: }
	
	# returns the outcome given the board
	def outcome(self, board):
	    if board.is_checkmate():
	        if board.turn:
	            return "black"
	        else:
	            return "white"
	    else:
	        return "draw"

	# calculate the total moves, given the board
	def moves(self, board):
	    return board.fullmove_number * 2 - 1 if board.turn else board.fullmove_number * 2 - 2

	# play one game of "random" chess 
	# i: index of game (if running multiple simulations)
	def play(self, i):
	    board = chess.Board()
	    j = 0 # j counts iteration of current game
	    while not board.is_game_over():
	    	move = random.choice(list(board.legal_moves))
	    	j += 1
	    	# insert here operations completed after each move
	        board.push(move)
	    return i, outcome(board), moves(board)

    # color is boolean for white/black
	def colormap_visits(self, color):
		if color:
			visits = np.reshape(white_data, (8, 8, 6))
		else:
			visits = np.reshape(black_data, (8, 8, 6))
		for i in range(6):
			plt.imshow

