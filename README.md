Test cases for GRPCURL: 
Run: 

Checks less than 3 verticies edge case:

grpcurl -plaintext -import-path /mnt/c/Users/qinyu/OneDrive/Desktop/Repos/Quad-RL/protos -proto drone.proto -d '{"vertex": [{"vertices": [{"x": 1.0, "y": 2.0, "z": 0.0}]}]}' 127.0.0.1:50051 dronecontrol.DroneService/SetEnvironment

grpcurl -plaintext \
  -import-path /mnt/c/Users/qinyu/OneDrive/Desktop/Repos/Quad-RL/protos \
  -proto drone.proto \
  -d '{"goal": {"x": 5.0, "y": 10.0, "z": 2.0}, "vertex": []}' \
  127.0.0.1:50051 dronecontrol.DroneService/SetEnvironment

