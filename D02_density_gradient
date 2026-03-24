import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('file.csv')


# Or define a custom function
def epsilon(del_x):
    C = 5.3
    ZD = 10
    ZA = 1200+40+10
    f = 200
    eps = del_x*(ZD+ZA+ZA-f)/(C*f*ZD)

    return eps

def normalized_epsilon(del_x, del_x_1):
    eps = epsilon(del_x)
    eps_max = epsilon(del_x_1)
    return eps / eps_max

df['result'] = df.iloc[:, 1].apply(my_function)

# Plot
plt.plot(df.iloc[:, 1], df['result'])
plt.xlabel('Second Column')
plt.ylabel('Result')
plt.title('Function Applied to Second Column')
plt.show()
