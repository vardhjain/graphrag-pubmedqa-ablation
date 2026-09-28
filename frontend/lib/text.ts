export const TRUNCATED_REASONING_MESSAGE =
  "The model's reasoning was cut off before it produced an answer.";

// Reasoning models (deepseek-r1 on the Ollama fallback) emit <think>...</think>.
// The backend usually strips it, but a truncated generation can leave an
// unterminated block, so strip defensively here too:
//  - paired <think>...</think> blocks
//  - an unterminated trailing <think>... (generation cut off mid-reasoning)
//  - a leading stray ...</think> whose opening tag was never emitted
export function stripThink(text: string): string {
  let out = text.replace(/<think>[\s\S]*?<\/think>/gi, "");
  out = out.replace(/<think>[\s\S]*$/i, "");
  const strayClose = out.search(/<\/think>/i);
  if (strayClose !== -1) out = out.slice(strayClose + "</think>".length);
  out = out.trim();
  return out.length > 0 ? out : TRUNCATED_REASONING_MESSAGE;
}
