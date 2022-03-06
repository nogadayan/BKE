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

my_agent = MyAgent(alpha=0.02, epsilon=0.3)
random_agent = RandomAgent()

print("Kies uit 1 van de volgende opties: \n 1. train een agent \n 2. speel tegen een getrainde agent \n 3. speel tegen een ander persoon \n 4. plot een validatie grafiek \n 5. uitleg over hyperparameters")

choice = input()

if choice == '1':
  train(my_agent, 3000)
  save(my_agent, 'MyAgent_3000')

if choice == '2':
  train(my_agent, 3000)
  save(my_agent, 'MyAgent_3000')
  my_agent = load('MyAgent_3000')
  my_agent.learning = False
  start(player_x=my_agent)
if choice == '3':
  start()
if choice == '4':
  train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=10000,
    trainings=100,
    validations=1000)
if choice == '5':
  print("Een hyperparameter is een parameter waarvan de waarde wordt gebruikt om het leerproces te beheersen. Een parameter is een grenswaarde. Bij hyperparameters kun je deze grenswaardes aanpassen, met een minimumwaarde van 0 en een maximumwaarde van 1. Hyperparameters hebben vaak namen van het griekse alfabet. De twee hyperparameters die hier werden gebruikt, zijn alpha en epsilon. De alpha is de leerfactor van de computer. Deze bepaalt hoe snel de agent over nieuwe kennis beheerst. Hoe hoger het getal is, des te sneller zal de agent oude kennis vervangen door nieuwe kennis. De epsilon is de verkenningsfactor van de computer. Deze bepaalt hoe vaak de computer nieuwe dingen zal proberen. Hoe hoger dit getal is, des te vaker zal de computer een willekeurige actie proberen in plaats van de best bekende zet. Door deze twee parameters telkens te veranderen, zal duidelijk worden hoe 'slim' de agent is.")