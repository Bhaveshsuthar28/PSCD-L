import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { useTheme } from "@/hooks/useTheme";
import { EmptyState, ErrorState } from "@/components/ui/StateFeedback";
import { Skeleton } from "@/components/ui/Skeleton";

interface EChartBoxProps {
  option: EChartsOption | null;
  height?: number;
  loading?: boolean;
  error?: string | null;
  onRetry?: () => void;
  emptyTitle?: string;
  emptyDescription?: string;
  isEmpty?: boolean;
}

export function EChartBox({
  option,
  height = 280,
  loading = false,
  error = null,
  onRetry,
  emptyTitle = "No chart data yet",
  emptyDescription = "Process a video to populate this visualization.",
  isEmpty = false,
}: EChartBoxProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    const el = containerRef.current;
    if (!el || loading || error || isEmpty || !option) {
      if (chartRef.current) {
        chartRef.current.dispose();
        chartRef.current = null;
      }
      return;
    }

    const instance = echarts.init(el);
    chartRef.current = instance;

    const ro = new ResizeObserver(() => {
      instance.resize();
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      instance.dispose();
      chartRef.current = null;
    };
  }, [loading, error, isEmpty, resolvedTheme]);

  useEffect(() => {
    const instance = chartRef.current;
    if (!instance || !option || loading || error || isEmpty) return;

    const isDark = resolvedTheme === "dark";
    const themedDefaults: EChartsOption = {
      backgroundColor: "transparent",
      textStyle: {
        color: isDark ? "#cbd5e1" : "#374151",
        fontFamily: "Inter, system-ui, sans-serif",
      },
      tooltip: {
        backgroundColor: isDark ? "#1e2633" : "#ffffff",
        borderColor: isDark ? "#374459" : "#e2e6ec",
        textStyle: {
          color: isDark ? "#f1f5f9" : "#111827",
          fontSize: 12,
        },
      },
      ...option,
    };

    instance.setOption(themedDefaults, true);
  }, [option, resolvedTheme, loading, error, isEmpty]);

  if (loading) {
    return <Skeleton className="w-full" style={{ height }} />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={onRetry} />;
  }

  if (isEmpty || !option) {
    return <EmptyState title={emptyTitle} description={emptyDescription} />;
  }

  return (
    <div
      ref={containerRef}
      style={{ height }}
      className="w-full"
      aria-label="Analytics chart"
    />
  );
}

export function getChartPalette(isDark: boolean) {
  return {
    axisLine: isDark ? "#334155" : "#d1d5db",
    splitLine: isDark ? "#1e293b" : "#f1f5f9",
    axisLabel: isDark ? "#94a3b8" : "#6b7280",
    legendText: isDark ? "#cbd5e1" : "#374151",
    primary: isDark ? "#60a5fa" : "#1d4ed8",
    success: isDark ? "#4ade80" : "#15803d",
    warning: isDark ? "#fbbf24" : "#b45309",
    danger: isDark ? "#f87171" : "#b91c1c",
    neutral: isDark ? "#94a3b8" : "#64748b",
  };
}
