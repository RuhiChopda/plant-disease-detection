import { Header } from "@/components/Header";
import { useScans } from "@/hooks/use-scans";
import { ScanCard } from "@/components/ScanCard";
import { Loader2, Search } from "lucide-react";
import { useState } from "react";

export default function History() {
  const { data: scans, isLoading } = useScans();
  const [search, setSearch] = useState("");

  const filteredScans = scans?.filter(scan => 
    scan.diseaseName?.toLowerCase().includes(search.toLowerCase()) ||
    scan.analysis?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-background leaf-pattern font-sans">
      <Header />
      
      <main className="container max-w-7xl mx-auto px-4 py-12">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-12 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div>
            <h1 className="text-4xl font-display font-bold text-foreground">Your Garden History</h1>
            <p className="text-muted-foreground mt-2">Track the health of your plants over time.</p>
          </div>
          
          <div className="relative w-full md:w-80">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input 
              type="text"
              placeholder="Search diagnoses..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-border bg-white/50 focus:bg-white focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all outline-none"
            />
          </div>
        </div>

        {isLoading ? (
          <div className="flex justify-center py-20">
            <Loader2 className="w-10 h-10 text-primary animate-spin" />
          </div>
        ) : filteredScans && filteredScans.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-in fade-in slide-in-from-bottom-8 duration-700">
            {filteredScans.map((scan) => (
              <ScanCard key={scan.id} scan={scan} />
            ))}
          </div>
        ) : (
          <div className="text-center py-20 bg-muted/30 rounded-3xl border border-dashed border-border animate-in fade-in zoom-in-95 duration-500">
            <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
              <Search className="w-8 h-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold text-foreground">No scans found</h3>
            <p className="text-muted-foreground mt-2">
              {search ? "Try adjusting your search terms" : "You haven't scanned any plants yet"}
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
