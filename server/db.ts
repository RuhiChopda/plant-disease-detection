import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import * as schema from "@shared/schema";

if (!process.env.DATABASE_URL) {
  throw new Error("DATABASE_URL is not set");
}

// DATABASE_URL = file:./dev.db
const dbFile = process.env.DATABASE_URL.replace("file:", "");

const sqlite = new Database(dbFile);
export const db = drizzle(sqlite, { schema });
