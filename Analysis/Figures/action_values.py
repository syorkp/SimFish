import matplotlib.pyplot as plt

values = [0.1, 3, 0.5, 9, 2, 0.5, 1.5]

plt.bar([i for i in range(len(values))], values, width=0.5)
plt.bar([3], values[3], width=0.5, color="r")
plt.axis(False)
plt.savefig("Panels/Panel-2/action_values.jpg")
plt.show()


