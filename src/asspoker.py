import treys 
from treys import Evaluator
from treys import Card
evaluator = Evaluator()

fullDeck = [f'{rank}{suit}' for rank in 'AKQJT98765432' for suit in 'shdc']

holeCards = [(input('Enter Hole Card 1:').strip(' ')).capitalize(), (input('Enter Hole Card 2:').strip(' ')).capitalize()]

hand = [Card.new(holeCards[0]), Card.new(holeCards[1])]

board = []


