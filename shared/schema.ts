import { pgTable, text, serial, integer, boolean, timestamp, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// === TABLE DEFINITIONS ===
export const scans = pgTable("scans", {
  id: serial("id").primaryKey(),
  imageUrl: text("image_url").notNull(),
  diseaseName: text("disease_name"),
  confidence: integer("confidence"), // 0-100
  analysis: text("analysis"), // Detailed analysis from AI
  prevention: text("prevention"), // Prevention steps from AI
  createdAt: timestamp("created_at").defaultNow(),
});

// === BASE SCHEMAS ===
export const insertScanSchema = createInsertSchema(scans).omit({ id: true, createdAt: true });

// === EXPLICIT API CONTRACT TYPES ===
export type Scan = typeof scans.$inferSelect;
export type InsertScan = z.infer<typeof insertScanSchema>;

// Request types
export type CreateScanRequest = {
  image: string; // Base64 or URL
};

// Response types
export type ScanResponse = Scan;
export type ScansListResponse = Scan[];
