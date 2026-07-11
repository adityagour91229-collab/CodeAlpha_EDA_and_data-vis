import express from "express";
import path from "path";
import dotenv from "dotenv";
import { GoogleGenAI, Type } from "@google/genai";
import { createServer as createViteServer } from "vite";

// Load environment variables
dotenv.config();

const app = express();
const PORT = 3000;

// Increase payload limits for datasets
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb", extended: true }));

// Initialize Gemini SDK with telemetry header
const apiKey = process.env.GEMINI_API_KEY;
let ai: GoogleGenAI | null = null;

if (apiKey) {
  ai = new GoogleGenAI({
    apiKey: apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
} else {
  console.warn("WARNING: GEMINI_API_KEY environment variable is not set.");
}

// API Route: Analyze dataset
app.post("/api/analyze", async (req, res) => {
  try {
    if (!ai) {
      return res.status(500).json({
        error: "Gemini API client is not initialized. Please ensure GEMINI_API_KEY is configured in your Secrets.",
      });
    }

    const { columns, rowCount, sampleRows, stats, customPrompt } = req.body;

    if (!columns || !Array.isArray(columns) || columns.length === 0) {
      return res.status(400).json({ error: "Invalid dataset metadata: 'columns' is required." });
    }

    // Construct a comprehensive prompt for Gemini
    const prompt = `
      You are an expert Chief Data Scientist and elite UI/UX Analytics storyteller.
      Analyze the following dataset metadata, summary statistics, and sample data.
      Then, generate a compelling executive summary, a deep data story, key anomalies/outliers/trends, and actionable business recommendations.

      --- DATASET OVERVIEW ---
      - Total Rows: ${rowCount || "Unknown"}
      - Columns: ${columns.join(", ")}

      --- STATISTICAL SUMMARY ---
      ${JSON.stringify(stats || {}, null, 2)}

      --- SAMPLE DATA (First ${sampleRows ? sampleRows.length : 0} rows) ---
      ${JSON.stringify(sampleRows || [], null, 2)}

      ${customPrompt ? `--- USER'S SPECIFIC ANALYSIS FOCUS --- \n${customPrompt}` : ""}

      Please generate:
      1. An executive summary (1-2 sentences, concise and professional).
      2. A data story (2 dynamic paragraphs highlighting hidden relationships, interesting findings, or overall patterns).
      3. A list of key anomalies, notable trends, or interesting outliers (with titles, explanations, and types).
      4. A list of 3-4 highly actionable business recommendations based on these insights.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: {
              type: Type.STRING,
              description: "A high-level executive summary of the dataset (1-2 sentences).",
            },
            dataStory: {
              type: Type.STRING,
              description: "A rich, narrative paragraph telling a story about the findings and relationships in the data.",
            },
            anomalies: {
              type: Type.ARRAY,
              description: "Significant anomalies, outliers, or specific interesting trends spotted in the data.",
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING, description: "Title of the anomaly or trend (e.g. 'Spike in Q3 Tech Hiring')" },
                  description: { type: Type.STRING, description: "Detailed description of what is happening and why it is important." },
                  type: { type: Type.STRING, description: "Must be one of: 'anomaly', 'trend', or 'outlier'." },
                },
                required: ["title", "description", "type"],
              },
            },
            recommendations: {
              type: Type.ARRAY,
              description: "Actionable strategic business recommendations based directly on the data analysis.",
              items: { type: Type.STRING },
            },
          },
          required: ["summary", "dataStory", "anomalies", "recommendations"],
        },
      },
    });

    const text = response.text;
    if (!text) {
      throw new Error("Empty response from Gemini API");
    }

    const result = JSON.parse(text.trim());
    return res.json(result);
  } catch (error: any) {
    console.error("Analysis Error:", error);
    return res.status(500).json({
      error: "Failed to analyze data using AI.",
      details: error.message || error,
    });
  }
});

// Configure Vite or Static asset serving
async function setupServer() {
  if (process.env.NODE_ENV !== "production") {
    console.log("Starting server in DEVELOPMENT mode with Vite middleware...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("Starting server in PRODUCTION mode with static files...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running at http://localhost:${PORT}`);
  });
}

setupServer().catch((err) => {
  console.error("Failed to start server:", err);
});
