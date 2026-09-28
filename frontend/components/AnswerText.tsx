// The LLM's answers come back as markdown-ish plain text (Gemini's prompt
// gives it no format instruction) -- **bold**, *italic*, "*   " bullet lists,
// "1. " numbered lists, occasional "## " headings, and paragraph breaks all
// render as literal characters if dumped into a whitespace-pre-wrap <div>.
// This is a small, dependency-free renderer for exactly those patterns, rather
// than pulling in a full markdown library. It never uses
// dangerouslySetInnerHTML: every piece of model text is rendered as a React
// text node.

import type { ReactNode } from "react";

function formatInline(text: string, keyPrefix: string): ReactNode[] {
  // **bold** first, then *italic* inside the remaining plain runs.
  const parts = text.split(/(\*\*[^*]+\*\*)/g).filter((p) => p.length > 0);
  return parts.flatMap((part, i) => {
    const key = `${keyPrefix}-${i}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return [<strong key={key}>{part.slice(2, -2)}</strong>];
    }
    // Single-asterisk italic: must hug non-space characters so "a * b" or a
    // stray "*" is left alone.
    const sub = part.split(/(\*[^*\s](?:[^*]*[^*\s])?\*)/g).filter((p) => p.length > 0);
    return sub.map((s, j) =>
      s.length > 2 && s.startsWith("*") && s.endsWith("*") ? (
        <em key={`${key}-${j}`}>{s.slice(1, -1)}</em>
      ) : (
        <span key={`${key}-${j}`}>{s}</span>
      )
    );
  });
}

// Ordered markers are 1-2 digits only, so a line opening with a year
// ("2019. The trial...") isn't mistaken for list item 2019.
const UNORDERED_MARKER = /^\s*[*-]\s+/;
const ORDERED_MARKER = /^\s*(\d{1,2})\.\s+/;
const HEADING = /^\s*#{1,6}\s+/;

type Segment =
  | { type: "list"; ordered: boolean; start: number; items: string[] }
  | { type: "heading"; text: string }
  | { type: "text"; lines: string[] };

// Walks all lines once. Consecutive list lines group into one list; a blank
// line between two items of the same list kind does NOT end the list (LLMs
// often emit "1. foo\n\n2. bar"), which previously produced several
// one-item <ol>s that each restarted at 1.
function segment(text: string): Segment[] {
  const segments: Segment[] = [];
  let blankSinceLast = false;

  for (const rawLine of text.split("\n")) {
    if (rawLine.trim().length === 0) {
      blankSinceLast = true;
      continue;
    }
    const last = segments[segments.length - 1];
    const ordMatch = rawLine.match(ORDERED_MARKER);
    const isUnordered = UNORDERED_MARKER.test(rawLine);

    if (HEADING.test(rawLine)) {
      segments.push({ type: "heading", text: rawLine.replace(HEADING, "").replace(/\s*#+\s*$/, "") });
    } else if (ordMatch || isUnordered) {
      const ordered = !!ordMatch;
      const stripped = ordered
        ? rawLine.replace(ORDERED_MARKER, "")
        : rawLine.replace(UNORDERED_MARKER, "");
      if (last && last.type === "list" && last.ordered === ordered) {
        last.items.push(stripped);
      } else {
        segments.push({
          type: "list",
          ordered,
          start: ordMatch ? parseInt(ordMatch[1], 10) : 1,
          items: [stripped],
        });
      }
    } else if (last && last.type === "text" && !blankSinceLast) {
      last.lines.push(rawLine);
    } else if (last && last.type === "list" && !blankSinceLast && /^\s+/.test(rawLine)) {
      // Indented continuation of the previous list item.
      last.items[last.items.length - 1] += ` ${rawLine.trim()}`;
    } else {
      segments.push({ type: "text", lines: [rawLine] });
    }
    blankSinceLast = false;
  }

  return segments;
}

export default function AnswerText({ text }: { text: string }) {
  const segments = segment(text.trim());

  return (
    <>
      {segments.map((seg, idx) => {
        const key = String(idx);
        const spacing = idx > 0 ? "mt-2" : undefined;
        if (seg.type === "heading") {
          return (
            <p key={key} className={`font-semibold ${spacing ?? ""}`}>
              {formatInline(seg.text, key)}
            </p>
          );
        }
        if (seg.type === "list") {
          const items = seg.items.map((item, ii) => (
            <li key={ii}>{formatInline(item, `${key}-${ii}`)}</li>
          ));
          return seg.ordered ? (
            <ol
              key={key}
              start={seg.start !== 1 ? seg.start : undefined}
              className="list-decimal ml-5 my-1 space-y-0.5"
            >
              {items}
            </ol>
          ) : (
            <ul key={key} className="list-disc ml-5 my-1 space-y-0.5">
              {items}
            </ul>
          );
        }
        return (
          <p key={key} className={spacing}>
            {seg.lines.map((line, li, arr) => (
              <span key={li}>
                {formatInline(line, `${key}-${li}`)}
                {li < arr.length - 1 && <br />}
              </span>
            ))}
          </p>
        );
      })}
    </>
  );
}
