import matplotlib.pyplot as plt
x = [1,2,3]
plt.subplot(211)
plt.plot(x, label="test1")
plt.plot([3,2,1], label="test2")
plt.legend(bbox_to_anchor=(0.5, 1.1), loc='upper center', ncol=2)
plt.show()
