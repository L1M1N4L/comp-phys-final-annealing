import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import os

def create_animation_gif(frames, filename='annealing_animation.gif', fps=10):
    """Create a simple GIF animation of the annealing process."""
    print("Creating animation GIF...")
    
    # Create a custom colormap for better visualization
    colors = plt.cm.viridis(np.linspace(0, 1, max(max(frame.flatten()) for frame in frames) + 1))
    cmap = ListedColormap(colors)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Annealing Process')
    
    # Create initial plot
    im = ax.imshow(frames[0], cmap=cmap, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Grain ID')
    
    def update(frame):
        im.set_array(frame)
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)
    
    # Save animation
    print("Saving animation as GIF...")
    anim.save(filename, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Animation saved as '{filename}'")

def create_animated_plot(frames, energy_history, avg_grain_size_history, temperature_history, 
                        filename='annealing_with_plots.gif', fps=10):
    """Create an animated visualization with plots showing system properties."""
    print("Creating animated visualization with plots...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3)
    
    # Main microstructure plot
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.viridis(np.linspace(0, 1, max(max(frame.flatten()) for frame in frames) + 1))
    cmap = ListedColormap(colors)
    im = ax1.imshow(frames[0], cmap=cmap, interpolation='nearest')
    ax1.set_title('Microstructure')
    plt.colorbar(im, ax=ax1, label='Grain ID')
    
    # Energy plot
    ax2 = fig.add_subplot(gs[0, 1])
    energy_line, = ax2.plot([], [], 'b-', label='Energy')
    ax2.set_title('System Energy')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Energy')
    ax2.grid(True)
    
    # Grain size plot
    ax3 = fig.add_subplot(gs[0, 2])
    size_line, = ax3.plot([], [], 'r-', label='Grain Size')
    temp_line, = ax3.plot([], [], 'g-', label='Temperature')
    ax3.set_title('Grain Size and Temperature')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Value')
    ax3.grid(True)
    ax3.legend()
    
    # Set up animation
    def update(frame_idx):
        # Update microstructure
        im.set_array(frames[frame_idx])
        
        # Update energy plot
        energy_line.set_data(range(len(energy_history[:frame_idx+1])), energy_history[:frame_idx+1])
        ax2.relim()
        ax2.autoscale_view()
        
        # Update grain size and temperature plots
        size_line.set_data(range(len(avg_grain_size_history[:frame_idx+1])), 
                          avg_grain_size_history[:frame_idx+1])
        temp_line.set_data(range(len(temperature_history[:frame_idx+1])), 
                          temperature_history[:frame_idx+1])
        ax3.relim()
        ax3.autoscale_view()
        
        return [im, energy_line, size_line, temp_line]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
    
    # Save animation
    print("Saving animation as GIF...")
    anim.save(filename, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"Animation with plots saved as '{filename}'")

def visualize_results(frames, energy_history, avg_grain_size_history, temperature_history):
    """Create and save both types of visualizations."""
    # Create frames directory if it doesn't exist
    if not os.path.exists('frames'):
        os.makedirs('frames')
    
    # Create simple animation
    create_animation_gif(frames, 'frames/annealing_animation.gif')
    
    # Create detailed visualization with plots
    create_animated_plot(frames, energy_history, avg_grain_size_history, temperature_history,
                        'frames/annealing_with_plots.gif') 