import random

from _ml import MLAgent, train, save, RandomAgent, load, train_and_plot, validate, plot_validation
from _core import is_winner, opponent, start

class MyAgent(MLAgent):
  def evaluate(self, board):
      if is_winner(board, self.symbol):
           reward = 1
      elif is_winner(board, opponent[self.symbol]):
          reward = -1
      else:
          reward = 0
      return reward
    

random.seed(1)

my_agent = MyAgent()

print("Kies uit 1 van de volgende opties: \n 1. train een agent \n 2. speel tegen een getrainde agent \n 3. speel tegen een ander persoon \n 4. plot een validatie grafiek")