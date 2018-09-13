import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


print("result",1 - sigmoid(0.35439181018290655))