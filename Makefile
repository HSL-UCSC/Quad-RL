.PHONY: protos


PYTHONPATH=$PWD:$PYTHONPATH python -m rl_policy.server

# Path to .proto file(s)
PROTO_DIR = ./protos
PROTO_FILES = $(wildcard $(PROTO_DIR)/*.proto)

# Output directories
OUT_DIR = ./src

# OUT_DIR = ./src/rl_policy
PYTHON_OUT = $(OUT_DIR)
GRPCLIB_OUT = $(OUT_DIR)
BETTERPROTO_OUT = $(OUT_DIR)/hyrl_api


run:
	python -m rl_policy.server

test_getDirectionField:
	python -m rl_policy.getDirectionField

test_getDirectionPath:
	python -m rl_policy.getDirectionPath

test_getTrajectory:
	python -m rl_policy.getTrajectorySim

# Command to generate all Python files from .proto
protos:
	mkdir -p $(OUT_DIR)/hyrl_api
	python -m grpc_tools.protoc \
	  -I$(PROTO_DIR) \
	  --python_out=$(OUT_DIR) \
	  --grpclib_python_out=$(OUT_DIR) \
		--python_betterproto_out=$(BETTERPROTO_OUT) \
	  $(PROTO_DIR)/hyrl_api/obstacle_avoidance.proto

# Clean generated files (optional)
clean:
	rm -f $(OUT_DIR)/drone_pb2.py $(OUT_DIR)/drone_grpc.py
