{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        # Default shell for just running the code (minimal setup)
        default = pkgs.mkShell {
          packages = with pkgs; [ 
            python313
            python313Packages.uv
            grpcurl
            protobuf
          ];

          shellHook = ''
            echo "🔧 Setting up minimal Python environment with uv..."

            # Add GCC library path to LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH
            
            # Add src to PATH
            export PYTHONPATH=$PWD/src:$PYTHONPATH
            
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

        # Dev shell geared toward development, leaving your system's Neovim in place
        dev = pkgs.mkShell {
          packages = with pkgs; [ 
            python313
            python313Packages.uv
            zsh
            black
            texliveFull
            grpcurl
            protobuf
          ];

          shellHook = ''
            echo "🔧 Setting up Python development environment with uv..."

            # Add GCC library path to LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH
            
            # Add src to PATH
            export PYTHONPATH=$PWD/src:$PYTHONPATH
            uv sync --extra dev
            
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
                uv pip install -e .
              else
                echo "⚠️ No pyproject.toml found, skipping dependency installation"
              fi
            fi

            # Start Zsh if not already the active shell
            if [ "$SHELL" != "$(command -v zsh)" ]; then
              export SHELL="$(command -v zsh)"
              exec zsh
            fi
          '';
        };

        # Full development environment with neovim included
        full = pkgs.mkShell {
          packages = with pkgs; [ 
            python313
            python313Packages.uv
            zsh
            neovim
            black
            texliveFull
            grpcurl
            protobuf
          ];

          shellHook = ''
            echo "🔧 Setting up comprehensive Python development environment with uv..."

            # Add GCC library path to LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH
            
            # Add src to PATH
            export PYTHONPATH=$PWD/src:$PYTHONPATH
            uv sync --extra dev
            
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
                uv pip install -e .
              else
                echo "⚠️ No pyproject.toml found, skipping dependency installation"
              fi
            fi

            # Start Zsh if not already the active shell
            if [ "$SHELL" != "$(command -v zsh)" ]; then
              export SHELL="$(command -v zsh)"
              exec zsh
            fi
          '';
        };
        
        # Alias to make run the same as default
        run = self.devShells.${pkgs.system}.default;
      });
    };
}
