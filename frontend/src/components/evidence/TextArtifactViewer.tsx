import { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Check,
  Copy,
  Download,
  FileJson,
  FileSpreadsheet,
  RefreshCw,
} from "lucide-react";
import { resolveApiUrl } from "@/api/client";
import { Button } from "@/components/ui/Button";
import type { EvidenceRecord } from "@/types/api";

export function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\r") {
      if (text[i + 1] === "\n") i += 1;
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (ch === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }

  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows.filter((r) => r.length > 1 || (r.length === 1 && r[0] !== ""));
}

const DEFAULT_ROW_LIMIT = 200;
const DEFAULT_TEXT_CHAR_LIMIT = 500_000;

export function TextArtifactViewer({ item }: { item: EvidenceRecord }) {
  const [content, setContent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadNonce, setReloadNonce] = useState(0);
  const [viewMode, setViewMode] = useState<"parsed" | "raw">("parsed");
  const [filterText, setFilterText] = useState("");
  const [showAllRows, setShowAllRows] = useState(false);
  const [showFullText, setShowFullText] = useState(false);
  const [copied, setCopied] = useState(false);

  const url = item.download_url ? resolveApiUrl(item.download_url) : "";
  const lowerName = item.original_filename.toLowerCase();
  const isCsv = item.mime_type.includes("csv") || lowerName.endsWith(".csv");
  const isJson = item.mime_type.includes("json") || lowerName.endsWith(".json");

  useEffect(() => {
    setViewMode("parsed");
    setFilterText("");
    setShowAllRows(false);
    setShowFullText(false);
    setCopied(false);

    if (!url) {
      setContent(null);
      setLoading(false);
      setError("Download URL is unavailable for this artifact.");
      return;
    }

    const controller = new AbortController();
    setLoading(true);
    setError(null);

    fetch(url, { signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Failed to load artifact (HTTP ${res.status}).`);
        }
        return res.text();
      })
      .then((text) => {
        setContent(text);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setContent(null);
        setLoading(false);
        setError(err instanceof Error ? err.message : "Failed to load file content.");
      });

    return () => controller.abort();
  }, [url, item.id, reloadNonce]);

  const parsedCsv = useMemo(() => {
    if (!isCsv || content === null) return null;
    const rows = parseCsvRows(content);
    if (rows.length === 0) return { headers: [] as string[], body: [] as string[][] };
    return { headers: rows[0], body: rows.slice(1) };
  }, [isCsv, content]);

  const filteredCsvRows = useMemo(() => {
    if (!parsedCsv) return [];
    const q = filterText.trim().toLowerCase();
    if (!q) return parsedCsv.body;
    return parsedCsv.body.filter((row) =>
      row.some((cell) => cell.toLowerCase().includes(q)),
    );
  }, [parsedCsv, filterText]);

  const formattedJson = useMemo(() => {
    if (!isJson || content === null) return null;
    try {
      const parsed = JSON.parse(content) as unknown;
      const pretty = JSON.stringify(parsed, null, 2);
      let summary = "JSON object";
      if (Array.isArray(parsed)) {
        summary = `${parsed.length} items`;
      } else if (parsed && typeof parsed === "object") {
        const keys = Object.keys(parsed);
        summary = `${keys.length} keys (${keys.slice(0, 4).join(", ")}${keys.length > 4 ? ", …" : ""})`;
      }
      return { text: pretty, summary, valid: true };
    } catch {
      return { text: content, summary: "Raw JSON text", valid: false };
    }
  }, [isJson, content]);

  const handleCopy = async () => {
    if (!content) return;
    try {
      await navigator.clipboard.writeText(
        isJson && viewMode === "parsed" && formattedJson ? formattedJson.text : content,
      );
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      // Ignore clipboard permission errors
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-[220px] items-center justify-center rounded-md border border-border bg-surface-muted p-6 text-xs text-muted-foreground">
        Loading {item.original_filename}…
      </div>
    );
  }

  if (error || content === null) {
    return (
      <div className="flex min-h-[200px] flex-col items-center justify-center gap-3 rounded-md border border-border bg-surface-muted p-6 text-center text-xs text-foreground">
        <AlertCircle className="h-5 w-5 text-status-warning" aria-hidden="true" />
        <p>{error ?? "Unable to load artifact content."}</p>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setReloadNonce((n) => n + 1)}
          >
            <RefreshCw className="h-3.5 w-3.5" aria-hidden="true" />
            Retry
          </Button>
          {url && (
            <a href={url} download={item.original_filename}>
              <Button variant="outline" size="sm">
                <Download className="h-3.5 w-3.5" aria-hidden="true" />
                Download
              </Button>
            </a>
          )}
        </div>
      </div>
    );
  }

  const displayedCsvRows = showAllRows
    ? filteredCsvRows
    : filteredCsvRows.slice(0, DEFAULT_ROW_LIMIT);

  const rawDisplayText =
    isJson && viewMode === "parsed" && formattedJson ? formattedJson.text : content;
  const isTruncatedText =
    !showFullText && rawDisplayText.length > DEFAULT_TEXT_CHAR_LIMIT;
  const visibleText = isTruncatedText
    ? rawDisplayText.slice(0, DEFAULT_TEXT_CHAR_LIMIT)
    : rawDisplayText;

  return (
    <div className="space-y-2.5 text-xs">
      <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border bg-surface-muted px-3 py-2">
        <div className="flex flex-wrap items-center gap-2 text-muted-foreground">
          {isCsv ? (
            <FileSpreadsheet className="h-4 w-4 text-primary" aria-hidden="true" />
          ) : (
            <FileJson className="h-4 w-4 text-primary" aria-hidden="true" />
          )}
          <span className="font-medium text-foreground">{item.original_filename}</span>
          {isCsv && parsedCsv && (
            <span className="tabular-nums">
              {parsedCsv.body.length} rows × {parsedCsv.headers.length} columns
            </span>
          )}
          {isJson && formattedJson && <span>{formattedJson.summary}</span>}
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {isCsv && viewMode === "parsed" && (
            <input
              type="search"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="Filter rows…"
              aria-label="Filter CSV rows"
              className="h-7 w-40 rounded border border-border bg-surface px-2 text-xs text-foreground"
            />
          )}
          {(isCsv || isJson) && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setViewMode((m) => (m === "parsed" ? "raw" : "parsed"))}
            >
              {viewMode === "parsed"
                ? "Raw View"
                : isCsv
                  ? "Table View"
                  : "Formatted View"}
            </Button>
          )}
          <Button variant="outline" size="sm" onClick={() => void handleCopy()}>
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-status-success" aria-hidden="true" />
                Copied
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" aria-hidden="true" />
                Copy
              </>
            )}
          </Button>
          {url && (
            <a href={url} download={item.original_filename}>
              <Button variant="outline" size="sm">
                <Download className="h-3.5 w-3.5" aria-hidden="true" />
                Download
              </Button>
            </a>
          )}
        </div>
      </div>

      {isCsv && viewMode === "parsed" && parsedCsv ? (
        <div className="space-y-2">
          <div className="max-h-[52vh] overflow-auto rounded-md border border-border bg-surface">
            <table className="w-full border-collapse text-left text-[11px]">
              <thead className="sticky top-0 z-10 border-b border-border bg-surface-muted text-muted-foreground">
                <tr>
                  <th className="px-2.5 py-2 font-medium tabular-nums">#</th>
                  {parsedCsv.headers.map((header, idx) => (
                    <th key={`${header}-${idx}`} className="whitespace-nowrap px-2.5 py-2 font-medium">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {displayedCsvRows.length === 0 ? (
                  <tr>
                    <td
                      colSpan={Math.max(1, parsedCsv.headers.length + 1)}
                      className="px-3 py-6 text-center text-muted-foreground"
                    >
                      No CSV rows match your filter.
                    </td>
                  </tr>
                ) : (
                  displayedCsvRows.map((row, rIdx) => (
                    <tr key={rIdx} className="hover:bg-surface-muted/50">
                      <td className="px-2.5 py-1.5 font-mono text-muted-foreground tabular-nums">
                        {rIdx + 1}
                      </td>
                      {parsedCsv.headers.map((_, cIdx) => (
                        <td
                          key={cIdx}
                          className="max-w-[260px] truncate whitespace-nowrap px-2.5 py-1.5 font-mono text-foreground"
                          title={row[cIdx] ?? ""}
                        >
                          {row[cIdx] ?? ""}
                        </td>
                      ))}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          {filteredCsvRows.length > DEFAULT_ROW_LIMIT && !showAllRows && (
            <div className="flex items-center justify-between text-[11px] text-muted-foreground">
              <span>
                Showing {DEFAULT_ROW_LIMIT} of {filteredCsvRows.length} rows
              </span>
              <Button variant="outline" size="sm" onClick={() => setShowAllRows(true)}>
                Show All {filteredCsvRows.length} Rows
              </Button>
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-2">
          <pre className="max-h-[52vh] overflow-auto rounded-md border border-border bg-surface p-3 font-mono text-[11px] leading-relaxed text-foreground whitespace-pre">
            {visibleText}
          </pre>
          {isTruncatedText && (
            <div className="flex items-center justify-between text-[11px] text-muted-foreground">
              <span>
                Showing first {DEFAULT_TEXT_CHAR_LIMIT.toLocaleString()} of{" "}
                {rawDisplayText.length.toLocaleString()} characters
              </span>
              <Button variant="outline" size="sm" onClick={() => setShowFullText(true)}>
                Load Full Content
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
