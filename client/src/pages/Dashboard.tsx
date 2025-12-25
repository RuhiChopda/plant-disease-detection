import { Header } from "@/components/Header";
import { useAnalytics } from "@/hooks/use-analytics";
import { Activity, Leaf, AlertTriangle, TrendingUp } from "lucide-react";
import { Loader2 } from "lucide-react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { Card } from "@/components/ui/card";

const COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#ec4899'];

export default function Dashboard() {
  const { stats, trends, isLoading } = useAnalytics();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <div className="flex justify-center items-center h-96">
          <Loader2 className="w-8 h-8 text-primary animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-primary/5">
      <Header />
      
      <main className="container max-w-7xl mx-auto px-4 py-12">
        {/* Page Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-display font-bold text-foreground mb-2">Analytics Dashboard</h1>
          <p className="text-muted-foreground">Track your plant health monitoring journey</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          {/* Total Scans */}
          <Card className="bg-white/50 dark:bg-white/5 border border-border/50 backdrop-blur-sm p-6 hover-elevate">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground mb-1">Total Scans</p>
                <p className="text-3xl font-bold text-foreground">{stats?.totalScans || 0}</p>
              </div>
              <Activity className="w-8 h-8 text-primary/60" />
            </div>
          </Card>

          {/* Healthy Plants */}
          <Card className="bg-white/50 dark:bg-white/5 border border-border/50 backdrop-blur-sm p-6 hover-elevate">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground mb-1">Healthy Plants</p>
                <p className="text-3xl font-bold text-green-600">{stats?.healthyPlants || 0}</p>
              </div>
              <Leaf className="w-8 h-8 text-green-600/60" />
            </div>
          </Card>

          {/* Diseased Plants */}
          <Card className="bg-white/50 dark:bg-white/5 border border-border/50 backdrop-blur-sm p-6 hover-elevate">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground mb-1">Diseased Plants</p>
                <p className="text-3xl font-bold text-red-600">{stats?.diseasedPlants || 0}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-600/60" />
            </div>
          </Card>

          {/* Avg Confidence */}
          <Card className="bg-white/50 dark:bg-white/5 border border-border/50 backdrop-blur-sm p-6 hover-elevate">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground mb-1">Avg Confidence</p>
                <p className="text-3xl font-bold text-primary">{Math.round(stats?.averageConfidence || 0)}%</p>
              </div>
              <TrendingUp className="w-8 h-8 text-primary/60" />
            </div>
          </Card>
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-12">
          {/* Scan Trends */}
          <Card className="lg:col-span-2 bg-white/50 dark:bg-white/5 border border-border/50 backdrop-blur-sm p-6">
            <h3 className="text-lg font-semibold text-foreground mb-6">Scan Trends</h3>
            {trends && trends.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trends}>
                  <CartesianGrid stroke="var(--tw-border-color)" strokeOpacity={0.1} />
                  <XAxis dataKey="date" stroke="var(--tw-muted-foreground)" style={{ fontSize: '12px' }} />
                  <YAxis stroke="var(--tw-muted-foreground)" style={{ fontSize: '12px' }} />
                  <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))' }} />
                  <Legend />
                  <Line type="monotone" dataKey="scans" stroke="var(--tw-primary)" strokeWidth={2} dot={{ fill: 'var(--tw-primary)' }} />
                  <Line type="monotone" dataKey="healthy" stroke="#22c55e" strokeWidth={2} dot={{ fill: '#22c55e' }} />
                  <Line type="monotone" dataKey="diseased" stroke="#ef4444" strokeWidth={2} dot={{ fill: '#ef4444' }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-60 flex items-center justify-center text-muted-foreground">
                No trend data available
              </div>
            )}
          </Card>

          {/* Top Diseases */}
          <Card className="bg-white/50 dark:bg-white/5 border border-border/50 backdrop-blur-sm p-6">
            <h3 className="text-lg font-semibold text-foreground mb-6">Top Diseases Detected</h3>
            {stats?.topDiseases && stats.topDiseases.length > 0 ? (
              <div className="space-y-4">
                {stats.topDiseases.slice(0, 5).map((disease, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium text-foreground">{disease.name}</span>
                      <span className="text-muted-foreground">{disease.count}</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                      <div 
                        className="bg-primary h-full rounded-full transition-all"
                        style={{ width: `${disease.percentage}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-48 flex items-center justify-center text-muted-foreground">
                No disease data yet
              </div>
            )}
          </Card>
        </div>

        {/* Health Status Distribution */}
        {stats && (
          <Card className="bg-white/50 dark:bg-white/5 border border-border/50 backdrop-blur-sm p-6">
            <h3 className="text-lg font-semibold text-foreground mb-6">Plant Health Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={[
                    { name: 'Healthy', value: stats.healthyPlants },
                    { name: 'Diseased', value: stats.diseasedPlants },
                  ]}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value, percent }) => `${name}: ${value} (${(percent * 100).toFixed(0)}%)`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  <Cell fill="#22c55e" />
                  <Cell fill="#ef4444" />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        )}
      </main>
    </div>
  );
}
