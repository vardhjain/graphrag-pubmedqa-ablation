// The LLM's answers come back as markdown-ish plain text (Gemini's prompt
// gives it no format instruction) -- **bold**, "*   " bullet lists, and
// paragraph breaks all render as literal characters if dumped into a
// whitespace-pre-wrap <div>. This is a small, dependency-free renderer for
// exactly the patterns actually observed in production, rather than pulling
// in a full markdown library for a handful of constructs.

function formatInline(text: string, keyPrefix: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter((p) => p.length > 0);
  return parts.map((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    return <span key={key}>{part}</span>;
  });
}

const LIST_MARKER = /^\s*(?:[*-]\s+|\d+\.\s+)/;

type Segment =
  | { type: "list"; ordered: boolean; items: string[] }
  | { type: "text"; lines: string[] };

// A paragraph "block" (lines separated by blank lines) often mixes a lead-in
// sentence with a list right after it on the very next line ("The study
// found:\n*   item one\n*   item two") -- a single newline, not a blank-line
// break. Requiring *every* line in the block to be a list line misses this
// entirely and dumps the raw "*   " markers as text, so instead this walks
// line by line and groups consecutive list/non-list runs into segments.
function segmentBlock(block: string): Segment[] {
  const lines = block.split("\n").filter((l) => l.trim().length > 0);
  const segments: Segment[] = [];

  for (const line of lines) {
    const isListLine = LIST_MARKER.test(line);
    const last = segments[segments.length - 1];

    if (isListLine) {
      const ordered = /^\s*\d+\./.test(line);
      const stripped = line.replace(LIST_MARKER, "");
      if (last && last.type === "list" && last.ordered === ordered) {
        last.items.push(stripped);
      } else {
        segments.push({ type: "list", ordered, items: [stripped] });
      }
    } else if (last && last.type === "text") {
      last.lines.push(line);
    } else {
      segments.push({ type: "text", lines: [line] });
    }
  }

  return segments;
}

export default function AnswerText({ text }: { text: string }) {
  const blocks = text.trim().split(/\n{2,}/);
  const segments = blocks.flatMap((block, bi) =>
    segmentBlock(block).map((seg, si) => ({ ...seg, key: `${bi}-${si}` }))
  );

  return (
    <>
      {segments.map((seg, idx) => {
        if (seg.type === "list") {
          const ListTag = seg.ordered ? "ol" : "ul";
          return (
            <ListTag
              key={seg.key}
              className={`${seg.ordered ? "list-decimal" : "list-disc"} ml-5 my-1 space-y-0.5`}
            >
              {seg.items.map((item, ii) => (
                <li key={ii}>{formatInline(item, `${seg.key}-${ii}`)}</li>
              ))}
            </ListTag>
          );
        }
        return (
          <p key={seg.key} className={idx > 0 ? "mt-2" : undefined}>
            {seg.lines.map((line, li, arr) => (
              <span key={li}>
                {formatInline(line, `${seg.key}-${li}`)}
                {li < arr.length - 1 && <br />}
              </span>
            ))}
          </p>
        );
      })}
    </>
  );
}
