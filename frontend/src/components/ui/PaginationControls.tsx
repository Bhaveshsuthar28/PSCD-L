import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/Button";
import type { PaginationMeta } from "@/types/api";

export function PaginationControls({
  pagination,
  onPageChange,
}: {
  pagination: PaginationMeta | undefined;
  onPageChange: (page: number) => void;
}) {
  if (!pagination || pagination.total_items === 0) return null;

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border px-4 py-3 text-xs text-muted-foreground">
      <span className="tabular-nums">
        Page {pagination.page} of {Math.max(1, pagination.total_pages)} ({pagination.total_items} total)
      </span>
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={!pagination.has_previous}
          onClick={() => onPageChange(pagination.page - 1)}
        >
          <ChevronLeft className="h-3.5 w-3.5" aria-hidden="true" />
          Previous
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={!pagination.has_next}
          onClick={() => onPageChange(pagination.page + 1)}
        >
          Next
          <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      </div>
    </div>
  );
}
