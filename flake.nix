{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [ 
            python313
          ] ++ (with pkgs.python313Packages; [
              uv
              pkgs.zsh
              pkgs.neovim
              pkgs.black
              pkgs.texliveFull
              pkgs.grpcurl
              pkgs.python3
              pkgs.protobuf
            ]);

          shellHook = ''
            echo "🔧 Setting up Python virtual environment with uv..."

            # Add GCC library path to LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH
            
            # Create venv if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "📦 No .venv found, creating with uv..."
              uv venv
            fi

            # Activate the venv
            if [ -f ".venv/bin/activate" ]; then
              source .venv/bin/activate
              echo "✅ Activated Python venv at .venv"
              python --version
              # Install dependencies from pyproject.toml if it exists
              if [ -f "pyproject.toml" ]; then
                echo "📦 Installing dependencies from pyproject.toml..."
                uv pip install .
              else
                echo "⚠️ No pyproject.toml found, skipping dependency installation"
              fi
            else
              echo "❌ Failed to activate venv: .venv/bin/activate not found"
            fi
          '';
        };
      });
    };
}
