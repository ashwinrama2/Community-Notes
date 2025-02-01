import matplotlib.pyplot as plt

# Function to read data from a file
def read_loss_file(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]

# Read data from each file
loss1 = read_loss_file('loss.txt')
loss2 = read_loss_file('loss2.txt')
loss6 = read_loss_file('loss6.txt')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(loss1, color='red', label='Loss with 1 embedded dimension')
plt.plot(loss2, color='green', label='Loss with 2 embedded dimensions')
plt.plot(loss6, color='blue', label='Loss with 6 embedded dimensions')

# Add labels, title, and legend
plt.xlabel('Training Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Loss Progression with Different Embedded Dimensions for Matrix Factorization', fontsize=18)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()