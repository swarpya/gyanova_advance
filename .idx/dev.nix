# To learn more about how to use Nix to configure your environment
# see: https://developers.google.com/idx/guides/customize-idx-env
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"
  # Use https://search.nixos.org/packages to find packages
  packages = [
    # pkgs.go
    pkgs.python311
    pkgs.python312Packages.conda
    pkgs.git
    pkgs.cmake
    pkgs.clang_18     # BitNet recommends Clang 18+
    pkgs.python3
    pkgs.gnumake
    pkgs.wget
    pkgs.python311Packages.pip
    pkgs.sudo
    pkgs.gcc        # The C compiler
    pkgs.gnumake    # Make tool
    pkgs.cmake      # CMake build system
    pkgs.zlib       # Common dependency
    # pkgs.nodejs_20
    # pkgs.nodePackages.nodemon
  ];
  # Sets environment variables in the workspace
  env = {
    CC = "gcc";
    CXX = "g++";
  };
  idx = {
    # Search for the extensions you want on https://open-vsx.org/ and use "publisher.id"
    extensions = [
      # "vscodevim.vim"
      "google.gemini-cli-vscode-ide-companion"
    ];
    # Enable previews
    previews = {
      enable = true;
      previews = {
        # web = {
        #   # Example: run "npm run dev" with PORT set to IDX's defined port for previews,
        #   # and show it in IDX's web preview panel
        #   command = ["npm" "run" "dev"];
        #   manager = "web";
        #   env = {
        #     # Environment variables to set for your server
        #     PORT = "$PORT";
        #   };
        # };
      };
    };
    # Workspace lifecycle hooks
    workspace = {
      # Runs when a workspace is first created
      onCreate = {
        # Example: install JS dependencies from NPM
        # npm-install = "npm install";
        # Open editors for the following files by default, if they exist:
        default.openFiles = [ ".idx/dev.nix" "README.md" ];
      };
      # Runs when the workspace is (re)started
      onStart = {
        # Example: start a background task to watch and re-build backend code
        # watch-backend = "npm run watch-backend";
      };
    };
  };
}
