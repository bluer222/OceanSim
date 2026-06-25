# 1. ALWAYS import and start SimulationApp first
from isaacsim import SimulationApp

# Configure your simulation (headless, cameras enabled, etc.)
simulation_app = SimulationApp({"headless": False})

# 2. NOW you can import isaacsim sensors or core extensions safely
from isaacsim.sensors import Camera
# Note: For older versions of Isaac Sim, use: from omni.isaac.sensor import Camera

print("Success! Camera module imported successfully.")

# Your simulation loop here...
# ...

# 3. Clean up at the end
simulation_app.close()
