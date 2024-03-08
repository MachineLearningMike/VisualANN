import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng()

mu = 0
beta = 1
count = 100

regression = np.arange(0, count, dtype=float)
var1 = rng.normal(mu, beta, count)
var2 = rng.normal(mu + mu*beta*0.1, beta*0.1, count)
data = regression + var1 + var2

plt.plot(data, 'ro')
plt.plot(regression, 'b')

plt.show()
