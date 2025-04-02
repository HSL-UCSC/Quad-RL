# HyRL Server
This is an RPC server...

## Dependency Installations
To get set up, first you need to install the nix Environment in your WSL:Ubuntu. Run this command from the nix environment website

```curl -L https://nixos.org/nix/install | sh -s -- --daemon```

## Setting up the Nix environment
Once you have succesfully installed your nix environment, run ```nix develop``` in your WSL terminal ensuring that you are in the repository top level when you do so. This should proceed to install all the nix environment and python dependencies required to run the RPC server. 

## How to startup the server
After succesfully entering the nix environment, simply start the server by running ```make run```. This will initialize the server and load the pretrained dqn models. 

## How to use the server
grpcurl -plaintext -import-path /mnt/c/Users/qinyu/OneDrive/Desktop/Repos/Quad-RL/protos -proto drone.proto -d '{"vertex": [{"vertices": [{"x": 1.0, "y": 2.0, "z": 0.0}]}]}' 127.0.0.1:50051 dronecontrol.DroneService/SetEnvironment

grpcurl -plaintext \
  -import-path /mnt/c/Users/qinyu/Desktop/Repositories/rl_policy/protos \
  -proto drone.proto \
  -d '{"goal": {"x": 5.0, "y": 10.0, "z": 2.0}, "vertex": []}' \
  127.0.0.1:50051 dronecontrol.DroneService/SetEnvironment

  grpcurl -plaintext -d '{"drone_state": {"x": 1.0, "y": 0.5, "z": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}}' 127.0.0.1:50051 dronecontrol.DroneService.GetDirection

grpcurl -plaintext   -import-path ./protos   -proto drone.proto   -d '{"drone_state": {"x": 1.5, "y": 0, "z": 15.0}}'   127.0.0.1:50051 dronecontrol.DroneService.GetDirection