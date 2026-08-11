#!/usr/bin/env swift
// Train FOSS-KI intent classifier — Apple CreateML
// Usage: swift train.swift [intents.json] [output.mlmodel]

import Foundation
import CreateML

let args = CommandLine.arguments
let inputPath = args.count > 1 ? args[1] : "intents.json"
let outputPath = args.count > 2 ? args[2] : "intent.mlmodel"

guard let data = FileManager.default.contents(atPath: inputPath),
      let items = try? JSONSerialization.jsonObject(with: data) as? [[String: String]] else {
    print("ERROR: Cannot read \(inputPath)")
    exit(1)
}

var texts: [String] = []
var labels: [String] = []
for item in items {
    if let t = item["text"], let l = item["label"] {
        texts.append(t)
        labels.append(l)
    }
}

let table = try MLDataTable(dictionary: ["text": texts, "label": labels])
let unique = Set(labels)
print("\(items.count) examples, \(unique.count) intents: \(unique.sorted().joined(separator: ", "))")

let model = try MLTextClassifier(trainingData: table, textColumn: "text", labelColumn: "label")
let acc = (1.0 - model.trainingMetrics.classificationError) * 100
print("Training accuracy: \(String(format: "%.1f", acc))%")

try model.write(to: URL(fileURLWithPath: outputPath))
print("Saved: \(outputPath)")
