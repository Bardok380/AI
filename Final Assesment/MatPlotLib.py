import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

# Create line plot
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Prime Numbers')

# Add  labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Graph w/ 10 Data Points')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()