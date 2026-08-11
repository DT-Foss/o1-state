// FOSS-KI NLU Engine — Apple NaturalLanguage.framework (native)
// No ObjC bridge. No pyobjc. Pure Swift on ANE.
//
// Compile: swiftc -O -target arm64-apple-macosx14.0 engine.swift -o nlu_engine
// Protocol: JSON lines on stdin → JSON lines on stdout

import Foundation
import NaturalLanguage

setlinebuf(stdout)

func log(_ msg: String) {
    fputs("[nlu] \(msg)\n", stderr)
}

func respond(_ dict: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: dict),
       let str = String(data: data, encoding: .utf8) {
        print(str)
    } else {
        print("{\"ok\":false,\"error\":\"json failed\"}")
    }
}

// ── State ────────────────────────────────────────────────────

var ctxEmb: [String: NLContextualEmbedding] = [:]
var wordEmb: [String: NLEmbedding] = [:]
var classifier: NLModel? = nil
let recognizer = NLLanguageRecognizer()
var embDim: Int = 0

// ── Init ─────────────────────────────────────────────────────

let modelDir = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "."

for lang in [NLLanguage.german, .english] {
    if let emb = NLContextualEmbedding(language: lang) {
        if emb.hasAvailableAssets {
            do {
                try emb.load()
                ctxEmb[lang.rawValue] = emb
                embDim = emb.dimension
                log("contextual \(lang.rawValue): \(emb.dimension)d")
            } catch {
                log("contextual \(lang.rawValue) failed: \(error)")
            }
        } else {
            log("contextual \(lang.rawValue): no assets")
        }
    }
    if let emb = NLEmbedding.wordEmbedding(for: lang) {
        wordEmb[lang.rawValue] = emb
        if embDim == 0 { embDim = emb.dimension }
        log("word \(lang.rawValue): \(emb.dimension)d")
    }
}

let mlmodelcPath = "\(modelDir)/intent.mlmodelc"
if FileManager.default.fileExists(atPath: mlmodelcPath) {
    classifier = try? NLModel(contentsOf: URL(fileURLWithPath: mlmodelcPath))
    if classifier != nil { log("classifier loaded") }
}

// ── NLU ──────────────────────────────────────────────────────

func detectLang(_ text: String) -> (String, Double) {
    recognizer.reset()
    recognizer.processString(text)
    let lang = recognizer.dominantLanguage ?? .english
    let hyp = recognizer.languageHypotheses(withMaximum: 1)
    return (lang.rawValue, hyp[lang] ?? 0.5)
}

func embedText(_ text: String) -> ([Float], Int, String)? {
    let (langCode, _) = detectLang(text)
    let lang = NLLanguage(rawValue: langCode)

    // Contextual (ANE-native, best quality)
    if let emb = ctxEmb[langCode] ?? ctxEmb["en"] {
        do {
            let result = try emb.embeddingResult(for: text, language: lang)
            let dim = emb.dimension
            var sum = [Float](repeating: 0, count: dim)
            var n: Float = 0
            result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { (vec: [Double], _: Range<String.Index>) -> Bool in
                for i in 0..<min(dim, vec.count) { sum[i] += Float(vec[i]) }
                n += 1
                return true
            }
            if n > 0 {
                let avg = sum.map { $0 / n }
                let norm = sqrt(avg.reduce(Float(0)) { $0 + $1 * $1 })
                return (norm > 0 ? avg.map { $0 / norm } : avg, dim, langCode)
            }
        } catch {}
    }

    // Word embedding fallback (average)
    if let we = wordEmb[langCode] ?? wordEmb["en"] {
        let words = text.lowercased().components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        let dim = we.dimension
        var sum = [Double](repeating: 0, count: dim)
        var n = 0
        for w in words {
            if let v = we.vector(for: w) {
                for i in 0..<dim { sum[i] += v[i] }
                n += 1
            }
        }
        if n > 0 {
            let avg = sum.map { Float($0 / Double(n)) }
            let norm = sqrt(avg.reduce(Float(0)) { $0 + $1 * $1 })
            return (norm > 0 ? avg.map { $0 / norm } : avg, dim, langCode)
        }
    }

    return nil
}

func classifyText(_ text: String) -> (String, Double, [String: Double])? {
    guard let model = classifier else { return nil }
    guard let label = model.predictedLabel(for: text) else { return nil }
    let hyp = model.predictedLabelHypotheses(for: text, maximumCount: 8)
    return (label, hyp[label] ?? 0.0, hyp)
}

func lemmatize(_ text: String) -> [[String: String]] {
    let (langCode, _) = detectLang(text)
    let lang = NLLanguage(rawValue: langCode)
    let tagger = NLTagger(tagSchemes: [.lemma, .lexicalClass])
    tagger.string = text
    tagger.setLanguage(lang, range: text.startIndex..<text.endIndex)

    var tokens: [[String: String]] = []
    tagger.enumerateTags(in: text.startIndex..<text.endIndex,
                         unit: .word, scheme: .lemma,
                         options: [.omitWhitespace, .omitPunctuation]) { tag, range in
        let word = String(text[range])
        let lemma = tag?.rawValue ?? word.lowercased()
        let posTag = tagger.tag(at: range.lowerBound, unit: .word, scheme: .lexicalClass)
        let pos = posTag.0?.rawValue ?? "Other"
        tokens.append(["text": word, "lemma": lemma, "pos": pos])
        return true
    }
    return tokens
}

// ── Ready ────────────────────────────────────────────────────

respond([
    "ready": true,
    "dim": embDim,
    "contextual": !ctxEmb.isEmpty,
    "languages": Array(Set(Array(ctxEmb.keys) + Array(wordEmb.keys))),
    "classifier": classifier != nil
])

// ── Loop ─────────────────────────────────────────────────────

while let line = readLine() {
    guard !line.isEmpty,
          let data = line.data(using: .utf8),
          let cmd = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let action = cmd["cmd"] as? String else {
        respond(["ok": false, "error": "parse error"])
        continue
    }

    switch action {

    case "embed":
        guard let text = cmd["text"] as? String, !text.isEmpty else {
            respond(["ok": false, "error": "missing text"])
            continue
        }
        if let (vec, dim, lang) = embedText(text) {
            let rounded = vec.map { Double(round($0 * 1e6) / 1e6) }
            respond(["ok": true, "vec": rounded, "dim": dim, "lang": lang])
        } else {
            respond(["ok": false, "error": "embedding failed"])
        }

    case "sim":
        guard let a = cmd["a"] as? String, let b = cmd["b"] as? String else {
            respond(["ok": false, "error": "missing a or b"])
            continue
        }
        if let (va, _, _) = embedText(a), let (vb, _, _) = embedText(b) {
            let dot = zip(va, vb).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            respond(["ok": true, "score": Double(round(dot * 1e6) / 1e6)])
        } else {
            respond(["ok": false, "error": "embedding failed"])
        }

    case "lang":
        guard let text = cmd["text"] as? String else {
            respond(["ok": false, "error": "missing text"])
            continue
        }
        let (lang, conf) = detectLang(text)
        respond(["ok": true, "lang": lang, "confidence": round(conf * 1000) / 1000])

    case "classify":
        guard let text = cmd["text"] as? String else {
            respond(["ok": false, "error": "missing text"])
            continue
        }
        if let (label, conf, scores) = classifyText(text) {
            respond(["ok": true, "label": label, "confidence": round(conf * 1000) / 1000, "scores": scores])
        } else {
            respond(["ok": false, "error": "no classifier"])
        }

    case "lemma":
        guard let text = cmd["text"] as? String else {
            respond(["ok": false, "error": "missing text"])
            continue
        }
        respond(["ok": true, "tokens": lemmatize(text)])

    case "info":
        respond([
            "ok": true, "dim": embDim,
            "contextual": !ctxEmb.isEmpty,
            "languages": Array(Set(Array(ctxEmb.keys) + Array(wordEmb.keys))),
            "classifier": classifier != nil,
            "version": "1.0"
        ])

    case "quit":
        respond(["ok": true])
        exit(0)

    default:
        respond(["ok": false, "error": "unknown: \(action)"])
    }
}
