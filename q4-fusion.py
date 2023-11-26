import pandas as pd
import numpy as np

class Apple():
    def __init__(self, percent, maturity) -> None:
        self.percent = percent
        self.maturity = maturity
        self.state = 0
        self.weight = 0

csv = pd.read_csv('p4-features.csv')
csv.sort_values('maturity', ascending=True, inplace=True)

percent = csv['percent'].tolist()
maturity = csv['maturity'].tolist()
apples = []
for i in range(len(percent)):
    apple = Apple(percent[i], maturity[i])
    apples.append(apple)

maxx = max(maturity)
minn = min(maturity)
low = (minn * 2 + maxx) / 3
high = (minn + maxx * 2) / 3

for apple in apples:
    if apple.maturity < low:
        apple.state = 0
    elif apple.maturity <= high:
        apple.state = 1
    else:
        apple.state = 2

weight_mu = [164.40, 166.30, 166.40]
weight_sigma = [i / 24 for i in weight_mu]
apples_state = [list(filter(lambda x: x.state == i, apples)) for i in (0,1,2)]
percents_state = [[apple.percent for apple in i] for i in apples_state]
percent_mu = [float(np.mean(i)) for i in percents_state]
percent_sigma = [float(np.std(i)) for i in percents_state]

for apple in apples:
    apple.weight = (apple.percent - percent_mu[apple.state]) / percent_sigma[apple.state]
    apple.weight = (apple.weight * weight_sigma[apple.state]) + weight_mu[apple.state]

weights = pd.DataFrame({'weight': [apple.weight for apple in apples]})
weights.to_csv('p4-weight.csv', index=False)