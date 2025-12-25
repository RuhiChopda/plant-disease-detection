import { useQuery } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { api } from "@shared/routes";

export function useAnalytics() {
  const statsQuery = useQuery({
    queryKey: ['/api/analytics/stats'],
    queryFn: async () => {
      const res = await apiRequest(api.analytics.stats.method, api.analytics.stats.path);
      return res.json();
    },
  });

  const trendsQuery = useQuery({
    queryKey: ['/api/analytics/trends'],
    queryFn: async () => {
      const res = await apiRequest(api.analytics.trends.method, api.analytics.trends.path);
      return res.json();
    },
  });

  return {
    stats: statsQuery.data,
    trends: trendsQuery.data,
    isLoading: statsQuery.isLoading || trendsQuery.isLoading,
  };
}
