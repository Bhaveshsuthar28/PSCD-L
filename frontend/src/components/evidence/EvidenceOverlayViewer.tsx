import { useState } from "react";
import { AlertCircle, Download, Eye, EyeOff } from "lucide-react";
import { resolveApiUrl } from "@/api/client";
import { TextArtifactViewer } from "@/components/evidence/TextArtifactViewer";
import { Button } from "@/components/ui/Button";
import type { EvidenceRecord } from "@/types/api";

interface EvidenceOverlayViewerProps {
  item: EvidenceRecord;
  showBboxToggle?: boolean;
  maxHeightClass?: string;
}

export function EvidenceOverlayViewer({
  item,
  showBboxToggle = true,
  maxHeightClass = "max-h-[420px]",
}: EvidenceOverlayViewerProps) {
  const [showOverlay, setShowOverlay] = useState(
    item.evidence_type === "original_frame" || item.evidence_type === "probability_map",
  );
  const [imgError, setImgError] = useState(false);

  if (!item.mime_type.startsWith("image/")) {
    return <TextArtifactViewer key={item.id} item={item} />;
  }

  const url = item.download_url ? resolveApiUrl(item.download_url) : "";
  const hasBbox =
    Array.isArray(item.bbox_xywh) &&
    item.bbox_xywh.length === 4 &&
    typeof item.width === "number" &&
    item.width > 0 &&
    typeof item.height === "number" &&
    item.height > 0;

  let clampedBox: { x: number; y: number; w: number; h: number } | null = null;
  if (hasBbox && item.bbox_xywh && item.width && item.height) {
    const [rawX, rawY, rawW, rawH] = item.bbox_xywh;
    const x1 = Math.max(0, Math.min(rawX, item.width));
    const y1 = Math.max(0, Math.min(rawY, item.height));
    const x2 = Math.max(x1, Math.min(rawX + rawW, item.width));
    const y2 = Math.max(y1, Math.min(rawY + rawH, item.height));
    if (x2 > x1 && y2 > y1) {
      clampedBox = { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
    }
  }

  return (
    <div className="space-y-2">
      <div className="relative flex items-center justify-center overflow-hidden rounded-md border border-border bg-black/90 p-2">
        {!url || imgError ? (
          <div className="flex flex-col items-center justify-center gap-2 py-12 text-xs text-muted-foreground">
            <AlertCircle className="h-5 w-5 text-status-warning" aria-hidden="true" />
            <span>Evidence image unavailable or failed to load.</span>
          </div>
        ) : (
          <div
            className="relative inline-block"
            style={
              item.width && item.height
                ? { aspectRatio: `${item.width} / ${item.height}` }
                : undefined
            }
          >
            <img
              src={url}
              alt={`${item.evidence_type} (${item.original_filename})`}
              onError={() => setImgError(true)}
              className={`block ${maxHeightClass} w-auto max-w-full object-contain`}
            />
            {showOverlay && clampedBox && item.width && item.height && (
              <svg
                viewBox={`0 0 ${item.width} ${item.height}`}
                preserveAspectRatio="xMidYMid meet"
                className="pointer-events-none absolute inset-0 h-full w-full"
                aria-label={`Candidate region bounding box [${clampedBox.x}, ${clampedBox.y}, ${clampedBox.w}, ${clampedBox.h}]`}
              >
                <rect
                  x={clampedBox.x}
                  y={clampedBox.y}
                  width={clampedBox.w}
                  height={clampedBox.h}
                  fill="none"
                  stroke="#22c55e"
                  strokeWidth={Math.max(2, Math.round(item.width / 240))}
                />
              </svg>
            )}
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
        <div className="flex flex-wrap items-center gap-3">
          {item.evidence_type === "probability_map" && (
            <span>
              Model spatial probability visualization (not ground-truth object segmentation).
            </span>
          )}
          {clampedBox && (
            <span className="font-mono">
              bbox=[{clampedBox.x}, {clampedBox.y}, {clampedBox.w}, {clampedBox.h}]
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {showBboxToggle && clampedBox && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowOverlay((v) => !v)}
            >
              {showOverlay ? (
                <>
                  <EyeOff className="h-3.5 w-3.5" aria-hidden="true" />
                  Hide BBox Overlay
                </>
              ) : (
                <>
                  <Eye className="h-3.5 w-3.5" aria-hidden="true" />
                  Show BBox Overlay
                </>
              )}
            </Button>
          )}
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
    </div>
  );
}
