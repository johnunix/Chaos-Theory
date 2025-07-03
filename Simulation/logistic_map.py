import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def logistic_map(r, x0, steps=100):
    x = np.zeros(steps)
    x[0] = x0
    for i in range(1, steps):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

# Initial parameters
r_init = 3.7
x0_init = 0.4
steps = 100

# Initial data
result = logistic_map(r_init, x0_init, steps)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)
line, = ax.plot(range(steps), result, marker='o', linestyle='-', markersize=3)
ax.set_title(f'Logistic Map Simulation (r={r_init}, x0={x0_init})')
ax.set_xlabel('Iteration')
ax.set_ylabel('x')
ax.grid(True)

# Slider axes
ax_r = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_x0 = plt.axes([0.15, 0.05, 0.65, 0.03])

# Create sliders
slider_r = Slider(ax_r, 'Growth Rate r', 0.0, 4.0, valinit=r_init, valstep=0.01)
slider_x0 = Slider(ax_x0, 'Initial x0', 0.0, 1.0, valinit=x0_init, valstep=0.01)

# Update function
def update(val):
    r = slider_r.val
    x0 = slider_x0.val
    ydata = logistic_map(r, x0, steps)
    line.set_ydata(ydata)
    ax.set_title(f'Logistic Map Simulation (r={r:.2f}, x0={x0:.2f})')
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_r.on_changed(update)
slider_x0.on_changed(update)

plt.show()
