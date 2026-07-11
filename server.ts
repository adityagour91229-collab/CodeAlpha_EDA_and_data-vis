import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: "50mb" }));

  // Initialize Gemini safely
  let ai: GoogleGenAI | null = null;
  if (process.env.GEMINI_API_KEY) {
    ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  } else {
    console.warn("WARNING: GEMINI_API_KEY is not defined in the environment. AI-driven features will be disabled.");
  }

  // API Routes
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", aiEnabled: !!ai });
  });

  // Phase 1: Guided Inquiry (generate tailored analytical questions)
  app.post("/api/gemini/questions", async (req, res) => {
    try {
      if (!ai) {
        return res.status(500).json({ error: "Gemini API key is missing. Please add it to Secrets in Settings." });
      }

      const { goals, columns, sampleData } = req.body;

      const prompt = `You are an expert Senior Data Scientist performing an Exploratory Data Analysis.
The user's analytical goals: ${JSON.stringify(goals)}.
Columns in the dataset:
${JSON.stringify(columns)}

And here is a sample of the data:
${JSON.stringify(sampleData)}

Generate 3 to 5 tailored, highly specific, and domain-relevant analytical questions to guide their exploratory analysis. Do NOT generate generic questions. Incorporate the columns and goals directly into the questions and rationales.`;

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
        config: {
          systemInstruction: "You are an expert full-stack data scientist who produces highly specific, insightful exploratory questions. Always format the response exactly matching the requested JSON schema. Ensure targetColumns contains exact matches of column names from the dataset.",
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                question: { type: Type.STRING, description: "A tailored, domain-specific analytical question." },
                rationale: { type: Type.STRING, description: "A concise description of why this question is highly relevant based on the dataset structure and goals." },
                targetColumns: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                  description: "List of columns from the dataset relevant to answering this question."
                }
              },
              required: ["question", "rationale", "targetColumns"]
            }
          }
        }
      });

      const text = response.text;
      if (!text) {
        throw new Error("No text response received from Gemini.");
      }

      const questions = JSON.parse(text.trim());
      return res.json({ questions });
    } catch (error: any) {
      console.error("Error in /api/gemini/questions:", error);
      return res.status(500).json({ error: error.message || "Failed to generate questions" });
    }
  });

  // Phase 5: Data Quality Health Report
  app.post("/api/gemini/health-report", async (req, res) => {
    try {
      if (!ai) {
        return res.status(500).json({ error: "Gemini API key is missing. Please add it to Secrets in Settings." });
      }

      const { columnsSummary, shape, issues, correlations } = req.body;

      const prompt = `You are a Senior Data Quality Auditor. Generate an executive Data Health Report for this dataset.
Dataset Overview:
- Rows: ${shape.rows}
- Columns: ${shape.cols}

Data Summary (columns, types, missing values, unique count):
${JSON.stringify(columnsSummary)}

Detected Statistical/Quality Issues:
${JSON.stringify(issues)}

Highly Collinear/Correlated Numerical Pairs:
${JSON.stringify(correlations)}

Assess the dataset health. Grade the dataset from A (perfectly clean, model-ready) to F (severe issues, needs major refactoring). Provide a high-level executive summary, a list of critical/warning findings with statistical impacts, and a checklist of actionable preprocessing recommendations.`;

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
        config: {
          systemInstruction: "You are an expert Data Quality Auditor. Provide objective, helpful, and highly insightful audits. Always return a valid JSON response matching the provided schema.",
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              grade: { type: Type.STRING, description: "An overall data quality letter grade (e.g., A+, B-, C, D, F)." },
              summary: { type: Type.STRING, description: "An elegant, comprehensive executive summary (2-3 paragraphs) of the data's health, strengths, and primary weaknesses." },
              findings: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    title: { type: Type.STRING, description: "Short title of the finding." },
                    description: { type: Type.STRING, description: "Clear explanation of what the statistical issue/pattern is." },
                    severity: { type: Type.STRING, description: "Must be either 'info', 'warning', or 'critical'." },
                    impact: { type: Type.STRING, description: "The mathematical or analytical impact of this issue on modeling or visualization." }
                  },
                  required: ["title", "description", "severity", "impact"]
                }
              },
              recommendations: {
                type: Type.ARRAY,
                items: { type: Type.STRING },
                description: "List of actionable preprocessing steps or corrections the user should take."
              }
            },
            required: ["grade", "summary", "findings", "recommendations"]
          }
        }
      });

      const text = response.text;
      if (!text) {
        throw new Error("No text response received from Gemini.");
      }

      const report = JSON.parse(text.trim());
      return res.json({ report });
    } catch (error: any) {
      console.error("Error in /api/gemini/health-report:", error);
      return res.status(500).json({ error: error.message || "Failed to generate data health report" });
    }
  });

  // Vite development vs production asset serving
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Server] Exploratory Data Analysis App running on http://localhost:${PORT}`);
  });
}

startServer();
