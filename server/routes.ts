import type { Express } from "express";
import type { Server } from "http";
import { storage } from "./storage";
import { api } from "@shared/routes";
import { z } from "zod";
import OpenAI from "openai";

// Initialize OpenAI client using Replit's integration env vars
const openai = new OpenAI({
  apiKey: process.env.AI_INTEGRATIONS_OPENAI_API_KEY,
  baseURL: process.env.AI_INTEGRATIONS_OPENAI_BASE_URL,
});

async function seedDatabase() {
  const scans = await storage.getScans();
  if (scans.length === 0) {
    console.log("Seeding database with sample scans...");
    await storage.createScan({
      imageUrl: "https://images.unsplash.com/photo-1596707328574-e35492d3f7f2?q=80&w=600&auto=format&fit=crop",
      diseaseName: "Healthy",
      confidence: 98,
      analysis: "The plant leaves appear vibrant green with no signs of discoloration, spots, or wilting. The texture looks smooth and healthy.",
      prevention: "Continue with regular watering and ensure adequate sunlight. Monitor for any changes in leaf color.",
    });
    await storage.createScan({
      imageUrl: "https://images.unsplash.com/photo-1615214044558-8687b1c3132d?q=80&w=600&auto=format&fit=crop",
      diseaseName: "Powdery Mildew",
      confidence: 85,
      analysis: "White, powdery spots are visible on the leaves, indicative of powdery mildew fungal infection.",
      prevention: "Improve air circulation around the plant. Remove infected leaves. Apply a fungicide or a mixture of baking soda and water.",
    });
    console.log("Database seeded!");
  }
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  // Scans API
  app.get(api.scans.list.path, async (req, res) => {
    const scans = await storage.getScans();
    res.json(scans);
  });

  app.get(api.scans.get.path, async (req, res) => {
    const scan = await storage.getScan(Number(req.params.id));
    if (!scan) {
      return res.status(404).json({ message: 'Scan not found' });
    }
    res.json(scan);
  });

  app.post(api.scans.create.path, async (req, res) => {
    try {
      const input = api.scans.create.input.parse(req.body);
      
      // Call OpenAI to analyze the image
      // Note: In a real scenario, valid base64 strings or URLs are needed.
      // If the input is a data URL (data:image/jpeg;base64,...), OpenAI handles it.
      
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `You are an expert plant pathologist. Analyze the provided image of a plant. 
            Identify if there is any disease. 
            Provide the output in JSON format with the following keys:
            - diseaseName: string (Name of the disease or "Healthy")
            - confidence: number (0-100)
            - analysis: string (Detailed explanation of findings)
            - prevention: string (Steps to prevent or cure, if applicable. If healthy, provide general care tips.)`
          },
          {
            role: "user",
            content: [
              { type: "text", text: "Analyze this plant image for diseases." },
              {
                type: "image_url",
                image_url: {
                  url: input.image, 
                },
              },
            ],
          },
        ],
        response_format: { type: "json_object" },
        max_tokens: 1000,
      });

      const result = JSON.parse(response.choices[0].message.content || "{}");

      // Save to database
      const scan = await storage.createScan({
        imageUrl: input.image, // In a real app, we'd upload to blob storage and save URL
        diseaseName: result.diseaseName || "Unknown",
        confidence: result.confidence || 0,
        analysis: result.analysis || "Could not analyze image.",
        prevention: result.prevention || "No prevention steps available.",
      });

      res.status(201).json(scan);
    } catch (err) {
      console.error("Analysis error:", err);
      if (err instanceof z.ZodError) {
        return res.status(400).json({
          message: err.errors[0].message,
          field: err.errors[0].path.join('.'),
        });
      }
      res.status(500).json({ message: "Failed to analyze image" });
    }
  });

  app.delete(api.scans.delete.path, async (req, res) => {
    const id = Number(req.params.id);
    const scan = await storage.getScan(id);
    if (!scan) {
      return res.status(404).json({ message: 'Scan not found' });
    }
    await storage.deleteScan(id);
    res.status(204).send();
  });
  
  // Initialize seeding
  seedDatabase().catch(console.error);

  return httpServer;
}
