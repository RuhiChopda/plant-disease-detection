import { useQuery } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { api } from "@shared/routes";

export function useAnalytics() {
  const statsQuery = useQuery({
    queryKey: ['/api/analytics/stats'],
    queryFn: async () => {
      return apiRequest(api.analytics.stats.path, { method: api.analytics.stats.method });
    },
  });

  const trendsQuery = useQuery({
    queryKey: ['/api/analytics/trends'],
    queryFn: async () => {
      return apiRequest(api.analytics.trends.path, { method: api.analytics.trends.method });
    },
  });

  return {
    stats: statsQuery.data,
    trends: trendsQuery.data,
    isLoading: statsQuery.isLoading || trendsQuery.isLoading,
  };
}
