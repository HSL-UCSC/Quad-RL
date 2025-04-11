.PHONY: protos

# Path to your .proto file(s)
PROTO_DIR = ./protos
PROTO_FILES = $(wildcard $(PROTO_DIR)/*.proto)

# Output directories
OUT_DIR = ./src/rl_policy
PYTHON_OUT = $(OUT_DIR)
GRPCLIB_OUT = $(OUT_DIR)
BETTERPROTO_OUT = $(OUT_DIR)

run:
	cd src && PYTHONPATH=$$PWD/rl_policy:$$PYTHONPATH python -m rl_policy.server

test:
	cd src && PYTHONPATH=$$PWD/rl_policy:$$PYTHONPATH python -m rl_policy.client_sim

test_path:
	cd src && PYTHONPATH=$$PWD/rl_policy:$$PYTHONPATH python -m rl_policy.client_sim_path

# Command to generate all Python files from .proto
protos:
	python -m grpc_tools.protoc \
		-I$(PROTO_DIR) \
		--python_out=$(PYTHON_OUT) \
		--grpclib_python_out=$(GRPCLIB_OUT) \
		--python_betterproto_out=$(BETTERPROTO_OUT) \
		$(PROTO_FILES)

# Clean generated files (optional)
clean:
	rm -f $(OUT_DIR)/drone_pb2.py $(OUT_DIR)/drone_grpc.py