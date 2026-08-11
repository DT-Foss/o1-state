#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== FOSS-KI NLU Build ==="

# Compile engine
echo "Compiling engine.swift..."
swiftc -O -target arm64-apple-macosx14.0 engine.swift -o nlu_engine
echo "  nlu_engine: $(du -h nlu_engine | cut -f1)"

# Train classifier
if [ -f intents.json ]; then
    echo "Training intent classifier..."
    swift train.swift intents.json intent.mlmodel
    echo "Compiling CoreML model..."
    if xcrun coremlcompiler compile intent.mlmodel . 2>/dev/null; then
        rm -f intent.mlmodel
    else
        # Compile via Swift/CoreML at runtime
        swift -e '
import CoreML
import Foundation
let src = URL(fileURLWithPath: "intent.mlmodel")
let compiled = try MLModel.compileModel(at: src)
let dest = URL(fileURLWithPath: "intent.mlmodelc")
if FileManager.default.fileExists(atPath: dest.path) {
    try FileManager.default.removeItem(at: dest)
}
try FileManager.default.copyItem(at: compiled, to: dest)
print("Compiled: intent.mlmodelc")
' 2>&1
        rm -f intent.mlmodel
    fi
    echo "  intent.mlmodelc ready"
fi

echo "=== Done ==="
echo "Test: echo '{\"cmd\":\"info\"}' | ./nlu_engine ."
