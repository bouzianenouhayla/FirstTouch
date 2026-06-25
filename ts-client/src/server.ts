import express, { Request, Response } from "express";
import path from "path";
import { fileURLToPath } from "url";
import { ask, health, saveProfile, type AskRequest, type UserProfile } from "./client.js";

const app = express();
const PORT = process.env.PORT ?? 3000;
const __dirname = path.dirname(fileURLToPath(import.meta.url));

app.use(express.json());
app.use(express.static(path.join(__dirname, "../public")));

app.get("/api/health", async (_req: Request, res: Response) => {
  const data = await health();
  res.json(data);
});

app.post("/api/ask", async (req: Request, res: Response) => {
  const payload = req.body as AskRequest;
  const result = await ask(payload);
  res.json(result);
});

app.post("/api/profile", async (req: Request, res: Response) => {
  const profile = req.body as UserProfile;
  const result = await saveProfile(profile);
  res.json(result);
});

app.listen(PORT, () => {
  console.log(`web running on http://localhost:${PORT}`);
});
