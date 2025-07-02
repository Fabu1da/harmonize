#!/usr/bin/env zsh
set -e
DIR="$(dirname -- "$(readlink -f -- "$0")")"
COMA_ROOT="/Users/fabu1da/Downloads/coma 3.0 ce v3"
classpath="$(printf %s: "$COMA_ROOT"/lib/**/*.jar)"
javac -classpath "$classpath" "$DIR"/**/*.java
cd "$COMA_ROOT"
java -classpath "$classpath":"$DIR" Main "$@"
