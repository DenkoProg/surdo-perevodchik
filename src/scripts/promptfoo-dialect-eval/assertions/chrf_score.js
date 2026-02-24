/**
 * chrF Score Assertion
 *
 * Computes character F-score (chrF) between model output and reference translation.
 * chrF is particularly well-suited for morphologically rich languages like Ukrainian.
 *
 * Reference: Popovic (2015) https://aclanthology.org/W15-3049
 * - n=1..6 character n-grams averaged
 * - beta=2 (recall-weighted, standard chrF configuration)
 *
 * Threshold recommendation: >= 0.45 for acceptable translation quality
 * (dialect divergence means perfect scores are not expected)
 *
 * Used as a javascript assertion in promptfoo:
 *   - type: javascript
 *     value: file://assertions/chrf_score.js
 *     threshold: 0.45
 */

module.exports = async function (output, context) {
  const reference = (context.vars || {}).reference;

  if (!reference) {
    return { pass: true, score: 1.0, reason: "No reference provided - skipping chrF check." };
  }

  if (!output || typeof output !== "string" || output.trim() === "") {
    return { pass: false, score: 0, reason: "Model output is empty." };
  }

  // Normalize: trim whitespace, collapse multiple spaces, lowercase
  const normalize = (s) => s.trim().toLowerCase().replace(/\s+/g, " ");
  const hyp = normalize(output);
  const ref = normalize(reference);

  // Extract character n-grams as a frequency map
  function getNgrams(text, n) {
    const ngrams = new Map();
    for (let i = 0; i <= text.length - n; i++) {
      const ng = text.slice(i, i + n);
      ngrams.set(ng, (ngrams.get(ng) || 0) + 1);
    }
    return ngrams;
  }

  // Compute F-score for a single n-gram order
  function chrfN(hyp, ref, n) {
    const hypNg = getNgrams(hyp, n);
    const refNg = getNgrams(ref, n);

    let matches = 0;
    for (const [ng, count] of hypNg) {
      matches += Math.min(count, refNg.get(ng) || 0);
    }

    const hypTotal = Math.max(hyp.length - n + 1, 0);
    const refTotal = Math.max(ref.length - n + 1, 0);

    const precision = hypTotal > 0 ? matches / hypTotal : 0;
    const recall = refTotal > 0 ? matches / refTotal : 0;

    if (precision + recall === 0) return 0;
    // beta=2 gives more weight to recall (standard chrF configuration)
    const beta = 2;
    return ((1 + beta * beta) * precision * recall) / (beta * beta * precision + recall);
  }

  // Average chrF over n=1..6
  const maxN = 6;
  let total = 0;
  for (let n = 1; n <= maxN; n++) {
    total += chrfN(hyp, ref, n);
  }
  const score = total / maxN;
  const roundedScore = Math.round(score * 1000) / 1000;

  return {
    pass: score >= 0.45,
    score: roundedScore,
    reason: `chrF score: ${roundedScore.toFixed(3)} (ref: ${ref.length} chars, hyp: ${hyp.length} chars)`,
  };
};
