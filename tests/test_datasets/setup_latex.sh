#!/bin/bash

# Save the current directory
ORIGINAL_DIR=$(pwd)

# Get the system TeX Live distribution paths
TEXLIVE_PATH=$(which latex | xargs dirname)
TEXLIVE_BASE=$(echo $TEXLIVE_PATH | sed 's|/bin/.*$||')

# Export essential TeX environment variables
export PATH=$TEXLIVE_PATH:$PATH
export TEXMFCNF=$TEXLIVE_BASE/texmf-dist/web2c:$TEXLIVE_BASE/texmf-config/web2c:$TEXMFCNF

# If you need to include Perl modules from system
export PERL5LIB=/usr/share/perl5:/usr/lib/perl5:$PERL5LIB

# Run the command with all arguments passed to this script
"$@"

# Return to the original directory
cd $ORIGINAL_DIR