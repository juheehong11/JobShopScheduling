import random

f = open("100.in", "w")

f.write(str(82) + "\n")

for i in range(1, 83):
  # get random numbers here
  #duration: int, 0 < d <= 60
  d = random.randint(20, 60) 
  #deadline: int, 0 < t <= 1440
  if (i % 3 == 0):
      t = random.randint(d, 480)
  elif (i % 3 == 1):
      t = random.randint(480, 960)
  else:
      t = random.randint(480, 1440)
  #profit: float, 0 < p < 100
  if (d < 40):
    if (d % 10 == 0):
        p = round(random.uniform(60, 99.999), 3)
    else:
        p = round(random.uniform(.001, 80), 3)
  else:
    if (d % 10 == 0):
        p = round(random.uniform(.001, 40), 3)
    else:
        p = round(random.uniform(20, 99.999), 3)
  f.write(str(i) + " " + str(t) + " " + str(d) + " " + str(p) + "\n")

f.close()
